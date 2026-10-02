//! The completion thread of a stream and the profiles both threads collect.
use super::*;

/// The completion thread of a stream: waits for each batch in submission order, adds its
/// timestamps to the profile and lends its staging bytes to the handler.
pub(super) struct Completion<'c> {
    pub(super) ctx: &'c GpuContext,
    /// Transfer readback: the queue and the `t_done` timeline to wait on.
    pub(super) xfer: Option<(Arc<TransferQueue>, Arc<Timeline>)>,
    pub(super) shared: Arc<Shared>,
    pub(super) layout: StagingLayout,
    pub(super) n_kernels: usize,
    /// Index of the last kernel's end query.
    pub(super) last_kernel_end: usize,
    pub(super) timed: bool,
    /// Wait for a batch by polling (`PollType::Poll`) until its staging map completes, never with
    /// `PollType::Wait`; on Metal. wgpu-hal 30's Metal `Device::wait` errors with
    /// `DeviceError::Lost` ("No active command buffers for fence value") when it runs while a
    /// `queue.submit` on another thread (the producer's next batch) is between `Fence::maintain`,
    /// which drops command buffers whose status is already `Completed` although their completion
    /// handler has not yet raised the fence value, and pushing its own command buffer: it then
    /// finds the fence below the value and no pending command buffer that will reach it.
    /// wgpu-core turns that into a lost device and destroys every buffer (which surfaced as
    /// "Buffer with 'pipeline.staging' label has been destroyed" from `abandon`'s unmap). A
    /// non-blocking poll never calls `Device::wait`.
    pub(super) poll_only: bool,
    /// Test hook: the delivery after this many more fails (then the hook clears).
    pub(super) fail_after: Option<u32>,
}

/// The completion thread's share of `PipelineStats::transfer_ms` (GPU ticks, host seconds).
#[derive(Default)]
pub(super) struct CompletionProfile {
    /// Kernel ticks, per `Kernels::names`.
    pub(super) ticks: Vec<u64>,
    /// Per kernel: some batch had an unwritten begin or end timestamp (`unwritten_stamp`).
    pub(super) ticks_bad: Vec<bool>,
    /// Some batch had an unwritten marker timestamp: `upload_copy`, `readback` and `idle` are
    /// meaningless (Metal apparently writes none for the empty marker passes).
    pub(super) markers_bad: bool,
    pub(super) upload_copy: u64,
    pub(super) readback: u64,
    pub(super) idle: u64,
    /// End marker of the last batch finished (ticks), for `idle`.
    pub(super) last_end: Option<u64>,
    /// K3t's ticks over the batches that ran it (`Job::partial`), how many did, and whether one
    /// had an unwritten timestamp.
    pub(super) trunc_ticks: u64,
    pub(super) trunc_batches: u32,
    pub(super) trunc_bad: bool,
    pub(super) wait: f64,
    pub(super) deliver: f64,
    /// When the last wait for the GPU returned, for `host_drain`.
    pub(super) last_wait: Option<Instant>,
    pub(super) fail_after: Option<u32>,
}

/// The producer's share of `PipelineStats::transfer_ms` (host seconds).
#[derive(Default)]
pub(super) struct ProducerProfile {
    pub(super) upload_wait: f64,
    pub(super) upload_write: f64,
    pub(super) submit: f64,
    pub(super) fill: f64,
}

impl Completion<'_> {
    /// Delivers every job; drops `handler` before returning.
    pub(super) fn run(mut self, jobs: mpsc::Receiver<Job>, mut handler: Box<Handler<'_>>) -> (anyhow::Result<()>, CompletionProfile) {
        let shared = self.shared.clone();
        let _abort = AbortOnPanic(&shared);
        let mut prof =
            CompletionProfile { ticks: vec![0; self.n_kernels], ticks_bad: vec![false; self.n_kernels], ..Default::default() };
        let r = self.drain(jobs, &mut *handler, &mut prof);
        drop(handler);
        if r.is_err() {
            shared.abort();
        }
        prof.fail_after = self.fail_after;
        (r, prof)
    }

    pub(super) fn drain(
        &mut self,
        jobs: mpsc::Receiver<Job>,
        handler: &mut Handler<'_>,
        prof: &mut CompletionProfile,
    ) -> anyhow::Result<()> {
        // Ends once the producer has dropped its sender and every job is taken.
        for job in jobs {
            if self.shared.aborted() {
                // The producer failed: deliver nothing more (`abandon` frees the slots).
                break;
            }
            let t = Instant::now();
            // The staging map's result, when the wait already took it.
            let mut mapped = None;
            match (&self.xfer, &job.mapped) {
                (Some((tq, t_done)), _) => {
                    tq.wait(t_done, job.seq)?;
                    // Deliver the upload buffers' map callbacks of completed submissions.
                    self.ctx.device.poll(wgpu::PollType::Poll).context("device poll")?;
                }
                (None, Some(rx)) if self.poll_only => {
                    // The map completes with the submission (the staging buffer's last use).
                    mapped = Some(loop {
                        self.ctx.device.poll(wgpu::PollType::Poll).context("device poll")?;
                        match rx.recv_timeout(std::time::Duration::from_micros(200)) {
                            Ok(r) => break r,
                            // The producer failed: deliver nothing more, as above.
                            Err(mpsc::RecvTimeoutError::Timeout) if self.shared.aborted() => return Ok(()),
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(mpsc::RecvTimeoutError::Disconnected) => anyhow::bail!("staging map callback dropped"),
                        }
                    });
                }
                (None, _) => {
                    let wait = wgpu::PollType::Wait { submission_index: Some(job.submission.clone()), timeout: None };
                    self.ctx.device.poll(wait).context("device poll")?;
                }
            }
            let done = Instant::now();
            prof.wait += (done - t).as_secs_f64();
            prof.last_wait = Some(done);
            let lease = self.lease(&job, mapped)?;
            if self.timed {
                self.add_timestamps(lease.bytes(), job.partial, prof);
            }
            if let Some(k) = &mut self.fail_after {
                if *k == 0 {
                    self.fail_after = None;
                    anyhow::bail!("injected delivery failure (test hook)");
                }
                *k -= 1;
            }
            let t = Instant::now();
            handler(lease, job.first, job.n, job.tag)?;
            prof.deliver += t.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// Lends out `job`'s staging bytes (its batch completed); `mapped`: the staging map's result
    /// if the wait already received it.
    pub(super) fn lease(&self, job: &Job, mapped: Option<Result<(), wgpu::BufferAsyncError>>) -> anyhow::Result<Lease> {
        let (view, ptr, len) = match &*job.staging {
            Staging::Wgpu(staging) => {
                let r = match mapped {
                    Some(r) => r,
                    None => {
                        let rx = job.mapped.as_ref().context("wgpu staging without a map request")?;
                        // The submission completed, so its map callback has run or is running
                        // (on whichever thread polled: the producer polls too).
                        rx.recv().context("staging map callback dropped")?
                    }
                };
                r.context("map staging buffer")?;
                let view = staging.get_mapped_range(..).map_err(|e| anyhow!("mapped range: {e}"))?;
                let (ptr, len) = (view.as_ptr(), view.len());
                (Some(view), ptr, len)
            }
            Staging::Host(staging) => {
                // SAFETY: the transfer queue's copy into it completed (waited for t_done >=
                // job.seq), and the slot's next copy is only submitted once the lease is dropped.
                let bytes = unsafe { staging.mapped() };
                (None, bytes.as_ptr(), bytes.len())
            }
        };
        self.shared.set(job.slot, SlotState::Leased);
        Ok(Lease { shared: self.shared.clone(), slot: job.slot, staging: job.staging.clone(), view, ptr, len })
    }

    /// Adds one batch's timestamps (the kernels' pairs, the two markers, then K3t's pair when the
    /// batch was `partial`) to the profile.
    pub(super) fn add_timestamps(&self, view: &[u8], partial: bool, prof: &mut CompletionProfile) {
        let nk = self.n_kernels;
        // Only 4-byte aligned in general (seqs_bytes(1) is not a multiple of 8).
        let t = self.layout.ts as usize;
        let pairs = nk + 1 + partial as usize;
        let stamps: Vec<u64> =
            view[t..t + pairs * 16].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        if partial {
            let (t0, t1) = (stamps[2 * nk + 2], stamps[2 * nk + 3]);
            prof.trunc_ticks += t1.saturating_sub(t0);
            prof.trunc_batches += 1;
            prof.trunc_bad |= unwritten_stamp(t0) || unwritten_stamp(t1);
        }
        for (k, acc) in prof.ticks.iter_mut().enumerate() {
            *acc += stamps[2 * k + 1].saturating_sub(stamps[2 * k]);
            prof.ticks_bad[k] |= unwritten_stamp(stamps[2 * k]) || unwritten_stamp(stamps[2 * k + 1]);
        }
        let (m0, m1) = (stamps[2 * nk], stamps[2 * nk + 1]);
        prof.markers_bad |= [m0, m1, stamps[0], stamps[self.last_kernel_end]].into_iter().any(unwritten_stamp);
        prof.upload_copy += stamps[0].saturating_sub(m0);
        prof.readback += m1.saturating_sub(stamps[self.last_kernel_end]);
        if let Some(end) = prof.last_end {
            prof.idle += m0.saturating_sub(end);
        }
        prof.last_end = Some(m1);
    }
}

/// A timestamp the GPU did not write: resolved as 0 (an M4 Pro printed a `gpu_upload_copy` of
/// about 1.9e6 ms, the absolute GPU clock minus a zero start marker: Metal apparently samples
/// nothing for the empty marker passes) or as Metal's `MTLCounterErrorValue` (all ones).
pub(super) fn unwritten_stamp(t: u64) -> bool {
    t == 0 || t == u64::MAX
}
