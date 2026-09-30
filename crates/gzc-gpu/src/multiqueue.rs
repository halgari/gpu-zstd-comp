//! Extra Vulkan queues through wgpu-hal hooks (speed-2 E3).
//!
//! wgpu 30 creates one queue per device (family 0, queue 0) and chains every submission behind
//! the previous one with a semaphore waited at TOP_OF_PIPE, so nothing submitted through one
//! `wgpu::Queue` ever overlaps. Here the `VkDevice` is created by hand with more queues (e.g. the
//! async-compute family, or a second queue of family 0), and each queue is wrapped in its own
//! `wgpu::Device` through `hal::vulkan::Adapter::device_from_raw` + `create_device_from_hal`: every
//! queue gets a complete `GpuContext` (its own pipelines and allocator) on the same `VkDevice`, so
//! the rest of the crate runs on any of them unchanged. The `VkDevice` is destroyed once the last
//! of the hal devices has dropped (`DeviceOwner`). Vulkan only.
use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context as _, anyhow};
use ash::vk;

use crate::context::{GpuContext, Prepared};

/// Destroys the shared `VkDevice` when the last hal device wrapping it has dropped.
struct DeviceOwner(ash::Device);

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // SAFETY: every hal device (and so every wgpu object) created on this VkDevice is gone.
        unsafe {
            let _ = self.0.device_wait_idle();
            self.0.destroy_device(None);
        }
    }
}

/// A queue family of the adapter: its index, flags and queue count.
#[derive(Clone, Copy, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub flags: vk::QueueFlags,
    pub count: u32,
}

/// Contexts sharing one `VkDevice`: `main` on family 0 queue 0 (what `GpuContext::new` would
/// give) and one per requested extra queue, in request order.
pub struct MultiQueue {
    pub extra: Vec<GpuContext>,
    pub main: GpuContext,
    pub families: Vec<QueueFamily>,
}

impl MultiQueue {
    /// Opens the adapter as `GpuContext::with_options(allow_subgroups, mappable)` does, with the
    /// extra `(family, queue index)` queues. Errors on a non-Vulkan adapter, an unknown family or
    /// an index beyond the family's queue count, or (0, 0) (that is `main`).
    pub fn open(allow_subgroups: bool, mappable: bool, extra: &[(u32, u32)]) -> anyhow::Result<Self> {
        let p = Prepared::new(allow_subgroups, mappable)?;
        // SAFETY: the hal adapter is only used while `p.adapter` lives (this function).
        let hal = unsafe { p.adapter.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| anyhow!("multi-queue needs the Vulkan backend"))?;
        let instance = hal.shared_instance().raw_instance();
        let phd = hal.raw_physical_device();
        // SAFETY: plain property query.
        let families: Vec<QueueFamily> = unsafe { instance.get_physical_device_queue_family_properties(phd) }
            .iter()
            .enumerate()
            .map(|(i, f)| QueueFamily { index: i as u32, flags: f.queue_flags, count: f.queue_count })
            .collect();
        let mut counts = BTreeMap::from([(0u32, 1u32)]);
        for &(f, i) in extra {
            anyhow::ensure!((f, i) != (0, 0), "queue (0, 0) is the main queue");
            let fam = families.get(f as usize).ok_or_else(|| anyhow!("no queue family {f}"))?;
            anyhow::ensure!(
                fam.flags.intersects(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER) && i < fam.count,
                "queue family {f} has no compute or transfer queue {i} ({:?}, {} queues)",
                fam.flags,
                fam.count
            );
            let c = counts.entry(f).or_insert(0);
            *c = (*c).max(i + 1);
        }

        let features = p.required_features;
        let limits = p.required_limits.clone();
        let hints = wgpu::MemoryHints::default();
        let exts = hal.required_device_extensions(features);
        let mut phd_features = hal.physical_device_features(&exts, features);
        // TEMP probe: GZC_PROBE_PRIO0 = the family-0 queues' priority (others 1.0).
        let p0: f32 = std::env::var("GZC_PROBE_PRIO0").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let prio: Vec<Vec<f32>> = counts.iter().map(|(&f, &n)| vec![if f == 0 { p0 } else { 1.0 }; n as usize]).collect();
        let infos: Vec<vk::DeviceQueueCreateInfo> = counts
            .iter()
            .zip(&prio)
            .map(|((&f, _), p)| vk::DeviceQueueCreateInfo::default().queue_family_index(f).queue_priorities(p))
            .collect();
        let ext_ptrs: Vec<*const std::ffi::c_char> = exts.iter().map(|e| e.as_ptr()).collect();
        let info = phd_features
            .add_to_device_create(vk::DeviceCreateInfo::default().queue_create_infos(&infos).enabled_extension_names(&ext_ptrs));
        // SAFETY: the create info is what `open_with_callback` builds, plus queues.
        let raw = unsafe { instance.create_device(phd, &info, None) }.context("vkCreateDevice")?;
        let owner = Arc::new(DeviceOwner(raw.clone()));

        let open = |family: u32, index: u32| -> anyhow::Result<GpuContext> {
            let guard = owner.clone();
            // SAFETY: `raw` was created from this adapter with `exts` and these features, and has
            // this queue; it stays valid until the last drop callback (DeviceOwner).
            let dev = unsafe {
                hal.device_from_raw(
                    raw.clone(),
                    Some(Box::new(move || drop(guard))),
                    &exts,
                    features,
                    &limits,
                    &hints,
                    family,
                    index,
                )
            }
            .map_err(|e| anyhow!("device_from_raw({family}, {index}): {e}"))?;
            // SAFETY: `dev` was opened from `p.adapter`'s hal adapter with `p.descriptor()`'s
            // features and limits.
            let (device, queue) =
                unsafe { p.adapter.create_device_from_hal(dev, &p.descriptor()) }.context("create_device_from_hal")?;
            Ok(p.context(device, queue))
        };
        let main = open(0, 0)?;
        let extra = extra.iter().map(|&(f, i)| open(f, i)).collect::<anyhow::Result<Vec<_>>>()?;
        drop(owner);
        Ok(Self { extra, main, families })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor::{BatchBuffers, GpuParams, Kernels};
    use gzc_core::block::chunk_file;
    use gzc_core::config::BLOCK_SIZE;
    use gzc_core::params::LVL9;
    use std::time::Instant;

    /// Blocks from every `stride`-th file (sorted paths) under `dir`, until `n` blocks.
    fn corpus_blocks(dir: &std::path::Path, n: usize, stride: usize) -> Vec<Vec<u8>> {
        fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("dds") || x.eq_ignore_ascii_case("nif")) {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(dir, &mut files);
        files.sort();
        let mut out = Vec::new();
        for f in files.iter().step_by(stride) {
            out.extend(chunk_file(&std::fs::read(f).unwrap()).into_iter().map(|b| b.data));
            if out.len() >= n {
                break;
            }
        }
        out.truncate(n);
        assert_eq!(out.len(), n, "corpus too small");
        out
    }

    struct Set {
        bufs: BatchBuffers,
        n: u32,
    }

    fn upload(ctx: &GpuContext, blocks: &[Vec<u8>]) -> Set {
        let n = blocks.len() as u32;
        let bufs = BatchBuffers::new(ctx, n, false, &LVL9);
        let refs: Vec<&[u8]> = blocks.iter().map(|b| b.as_slice()).collect();
        ctx.queue.write_buffer(&bufs.data, 0, bytemuck::cast_slice(&crate::context::pack_blocks(&refs)));
        Set { bufs, n }
    }

    fn wait(ctx: &GpuContext) {
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    fn k3(ctx: &GpuContext, k: &Kernels, s: &Set) -> wgpu::CommandBuffer {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        k.record_parse(ctx, &mut enc, &s.bufs, s.n, None);
        enc.finish()
    }

    fn k12(ctx: &GpuContext, k: &Kernels, s: &Set) -> wgpu::CommandBuffer {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        k.record_best(ctx, &mut enc, &s.bufs, s.n, None);
        enc.finish()
    }

    struct Ts {
        set: wgpu::QuerySet,
        resolve: wgpu::Buffer,
    }

    fn ts(ctx: &GpuContext) -> Ts {
        let set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 4,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ts { set, resolve }
    }

    fn k3_ts(ctx: &GpuContext, k: &Kernels, s: &Set, t: &Ts) -> wgpu::CommandBuffer {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        let w = wgpu::ComputePassTimestampWrites {
            query_set: &t.set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        };
        k.record_parse(ctx, &mut enc, &s.bufs, s.n, Some(w));
        enc.resolve_query_set(&t.set, 0..2, &t.resolve, 0);
        enc.finish()
    }

    fn k12_ts(ctx: &GpuContext, k: &Kernels, s: &Set, t: &Ts) -> wgpu::CommandBuffer {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        k.record_best(ctx, &mut enc, &s.bufs, s.n, Some(&t.set));
        enc.resolve_query_set(&t.set, 0..4, &t.resolve, 0);
        enc.finish()
    }

    /// Prints the concurrent run's kernel intervals (ms from the first begin).
    fn timeline(main: &GpuContext, q: &GpuContext, ta: &Ts, tb: &Ts, label: &str) {
        let a: Vec<u64> = main.read_buffer(&ta.resolve, 0, 2);
        let b: Vec<u64> = q.read_buffer(&tb.resolve, 0, 4);
        let t0 = a[0].min(b[0]);
        let p = main.queue.get_timestamp_period() as f64 / 1e6;
        let ms = |t: u64| (t - t0) as f64 * p;
        eprintln!(
            "  {label}: K3(A) {:.1}..{:.1}  K1(B) {:.1}..{:.1}  K2(B) {:.1}..{:.1}",
            ms(a[0]),
            ms(a[1]),
            ms(b[0]),
            ms(b[1]),
            ms(b[2]),
            ms(b[3])
        );
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// E3 overlap probe (throwaway measurement, `--ignored`): K3 of set A on the main queue
    /// against K1+K2 of set B on another queue (the async-compute family, a second family-0 queue),
    /// alone and concurrently, plus the single-queue variant (K3(A) and K1(B) recorded without a
    /// barrier between them). `GZC_PROBE_N` blocks per set (default 1280), corpus from
    /// `GZC_PROBE_INPUT` (default data/corpus).
    #[test]
    #[ignore]
    fn overlap_probe() {
        let n: usize = std::env::var("GZC_PROBE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1280);
        let dir = std::env::var("GZC_PROBE_INPUT").unwrap_or_else(|_| "../../../../data/corpus".into());
        let dir = std::path::PathBuf::from(dir);
        let dir = if dir.exists() { dir } else { "/home/tbaldrid/oss/gpu-zstd-comp/data/corpus".into() };
        let reps: usize = std::env::var("GZC_PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
        let blocks = corpus_blocks(&dir, 2 * n, 7);
        let (a_blocks, b_blocks) = blocks.split_at(n);
        let _ = BLOCK_SIZE;

        let async_family = {
            let mq = MultiQueue::open(true, false, &[]).unwrap();
            for f in &mq.families {
                eprintln!("queue family {}: {:?} x{}", f.index, f.flags, f.count);
            }
            mq.families
                .iter()
                .find(|f| f.flags.contains(vk::QueueFlags::COMPUTE) && !f.flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|f| f.index)
        };
        let mut extra = vec![(0u32, 1u32)];
        if let Some(f) = async_family {
            extra.push((f, 0));
        }
        let mq = MultiQueue::open(true, false, &extra).unwrap();
        let params = GpuParams { matching: LVL9, emit_frames: false, huffman: true };
        // GZC_PROBE_SWAP=1: K3 runs on the last extra queue (the async family), the others on main.
        let swap = std::env::var("GZC_PROBE_SWAP").is_ok_and(|v| v == "1");
        let last = mq.extra.len() - 1;
        let main = if swap { &mq.extra[last] } else { &mq.main };
        let others: Vec<(&GpuContext, (u32, u32))> = if swap {
            vec![(&mq.main, (0, 0))]
        } else {
            mq.extra.iter().zip(extra.iter().copied()).collect()
        };
        let km = Kernels::new(main, params).unwrap();
        eprintln!("k3 mode {:?}, n = {n}", km.k3_mode());
        let a = upload(main, a_blocks);
        let bm = upload(main, b_blocks);
        main.queue.submit([k12(main, &km, &a)]);
        wait(main);

        // (a) K3(A) alone; (b) K1+K2(B) alone on main; single-queue no-barrier K3(A) + K1+K2(B).
        let time = |f: &mut dyn FnMut()| -> f64 {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        };
        let mut t_a = Vec::new();
        let mut t_b = Vec::new();
        let mut t_single = Vec::new();
        for _ in 0..reps {
            t_a.push(time(&mut || {
                main.queue.submit([k3(main, &km, &a)]);
                wait(main);
            }));
            t_b.push(time(&mut || {
                main.queue.submit([k12(main, &km, &bm)]);
                wait(main);
            }));
            t_single.push(time(&mut || {
                let mut enc = main.device.create_command_encoder(&Default::default());
                km.record_parse(main, &mut enc, &a.bufs, a.n, None);
                km.record_best(main, &mut enc, &bm.bufs, bm.n, None);
                main.queue.submit([enc.finish()]);
                wait(main);
            }));
        }
        let (ma, mb) = (median(t_a), median(t_b));
        eprintln!("K3(A) alone {ma:.2} ms, K1+K2(B) alone {mb:.2} ms, sum {:.2}", ma + mb);
        eprintln!(
            "single queue, K3(A) then K1(B) without a barrier, K2(B) after one: {:.2} ms ({:.2} x sum)",
            median(t_single.clone()),
            median(t_single) / (ma + mb)
        );

        for &(q, (f, i)) in &others {
            let kq = Kernels::new(q, params).unwrap();
            let b = upload(q, b_blocks);
            q.queue.submit([k12(q, &kq, &b)]);
            wait(q);
            let mut t_bq = Vec::new();
            let mut t_both = Vec::new();
            let mut t_both_rev = Vec::new();
            for _ in 0..reps {
                t_bq.push(time(&mut || {
                    q.queue.submit([k12(q, &kq, &b)]);
                    wait(q);
                }));
                let (mut ca, mut cb) = (Some(k3(main, &km, &a)), Some(k12(q, &kq, &b)));
                t_both.push(time(&mut || {
                    main.queue.submit(ca.take());
                    q.queue.submit(cb.take());
                    wait(main);
                    wait(q);
                }));
                let (mut ca, mut cb) = (Some(k3(main, &km, &a)), Some(k12(q, &kq, &b)));
                t_both_rev.push(time(&mut || {
                    q.queue.submit(cb.take());
                    main.queue.submit(ca.take());
                    wait(main);
                    wait(q);
                }));
            }
            let (ta, tb) = (ts(main), ts(q));
            for first_b in [false, true, false, true] {
                let (ca, cb) = (k3_ts(main, &km, &a, &ta), k12_ts(q, &kq, &b, &tb));
                if first_b {
                    q.queue.submit([cb]);
                    main.queue.submit([ca]);
                } else {
                    main.queue.submit([ca]);
                    q.queue.submit([cb]);
                }
                wait(main);
                wait(q);
                timeline(main, q, &ta, &tb, if first_b { "B first" } else { "A first" });
            }
            for (label, alone_main) in [("alone K3(A)", true), ("alone K1 / K2 (B)", false)] {
                if alone_main {
                    main.queue.submit([k3_ts(main, &km, &a, &ta)]);
                    wait(main);
                } else {
                    q.queue.submit([k12_ts(q, &kq, &b, &tb)]);
                    wait(q);
                }
                let v: Vec<u64> = if alone_main { main.read_buffer(&ta.resolve, 0, 2) } else { q.read_buffer(&tb.resolve, 0, 4) };
                let p = main.queue.get_timestamp_period() as f64 / 1e6;
                let d: Vec<String> = v.chunks(2).map(|c| format!("{:.2}", (c[1] - c[0]) as f64 * p)).collect();
                eprintln!("  {label}: {}", d.join(" / "));
            }
            // K3(A) against K1(B) alone and K2(B) alone (pred(B) is in place from the runs above).
            for mode in 0..3u32 {
                let mk = || {
                    let mut enc = q.device.create_command_encoder(&Default::default());
                    if mode == 2 {
                        kq.record_k1_or_k2(q, &mut enc, &b.bufs, b.n, true);
                    }
                    kq.record_k1_or_k2(q, &mut enc, &b.bufs, b.n, mode == 1);
                    enc.finish()
                };
                let (mut alone, mut af, mut bf) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..reps {
                    let mut cb = Some(mk());
                    alone.push(time(&mut || {
                        q.queue.submit(cb.take());
                        wait(q);
                    }));
                    let (mut ca, mut cb) = (Some(k3(main, &km, &a)), Some(mk()));
                    af.push(time(&mut || {
                        main.queue.submit(ca.take());
                        q.queue.submit(cb.take());
                        wait(main);
                        wait(q);
                    }));
                    let (mut ca, mut cb) = (Some(k3(main, &km, &a)), Some(mk()));
                    bf.push(time(&mut || {
                        q.queue.submit(cb.take());
                        main.queue.submit(ca.take());
                        wait(main);
                        wait(q);
                    }));
                }
                let (al, af, bf) = (median(alone), median(af), median(bf));
                eprintln!(
                    "queue ({f}, {i}): {} alone {al:.2} ms; with K3(A): A first {af:.2} ({:.2} x sum), B first {bf:.2} ({:.2} x sum)",
                    ["K1(B)", "K2(B)", "K2(B) then K1(B)"][mode as usize],
                    af / (ma + al),
                    bf / (ma + al)
                );
            }
            let mbq = median(t_bq);
            let (both, rev) = (median(t_both), median(t_both_rev));
            eprintln!(
                "queue ({f}, {i}): K1+K2(B) alone {mbq:.2} ms; K3(A) on main + K1+K2(B) here: {both:.2} ms \
                 ({:.2} x sum), submitted B first: {rev:.2} ms ({:.2} x sum)",
                both / (ma + mbq),
                rev / (ma + mbq)
            );
        }
    }

    /// E3 transfer-queue probe (`--ignored`): a frames-sized device -> host copy (the readback) on
    /// the dedicated transfer family, alone and concurrently with K3 and with K1+K2 on the main
    /// queue. Same env as `overlap_probe`.
    #[test]
    #[ignore]
    fn transfer_probe() {
        let n: usize = std::env::var("GZC_PROBE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1280);
        let reps: usize = std::env::var("GZC_PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
        let dir = std::path::PathBuf::from("/home/tbaldrid/oss/gpu-zstd-comp/data/corpus");
        let blocks = corpus_blocks(&dir, 2 * n, 7);
        let (a_blocks, b_blocks) = blocks.split_at(n);
        let fam = {
            let mq = MultiQueue::open(true, false, &[]).unwrap();
            mq.families
                .iter()
                .find(|f| f.flags.contains(vk::QueueFlags::TRANSFER) && !f.flags.contains(vk::QueueFlags::COMPUTE))
                .map(|f| f.index)
                .expect("no dedicated transfer family")
        };
        let mq = MultiQueue::open(true, false, &[(fam, 0)]).unwrap();
        let (main, t) = (&mq.main, &mq.extra[0]);
        let params = GpuParams { matching: LVL9, emit_frames: false, huffman: true };
        let km = Kernels::new(main, params).unwrap();
        let (a, b) = (upload(main, a_blocks), upload(main, b_blocks));
        main.queue.submit([k12(main, &km, &a)]);
        wait(main);
        // The readback: frames_bytes(n) from device-local memory into a mappable buffer.
        let size = crate::compressor::frames_bytes(n as u32);
        let src = t.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dst = t.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let copy = |ctx: &GpuContext, s: &wgpu::Buffer, d: &wgpu::Buffer| {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(s, 0, d, 0, size);
            enc.finish()
        };
        // The same copy on the main queue, for its serial cost.
        let msrc = main.storage_buffer("src", size, true);
        let mdst = main.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let time = |f: &mut dyn FnMut()| -> f64 {
            let tt = Instant::now();
            f();
            tt.elapsed().as_secs_f64() * 1e3
        };
        let mut r: Vec<Vec<f64>> = vec![Vec::new(); 7];
        for _ in 0..reps {
            let mut c = Some(copy(t, &src, &dst));
            r[0].push(time(&mut || {
                t.queue.submit(c.take());
                wait(t);
            }));
            let mut c = Some(copy(main, &msrc, &mdst));
            r[1].push(time(&mut || {
                main.queue.submit(c.take());
                wait(main);
            }));
            let mut k = Some(k3(main, &km, &a));
            r[2].push(time(&mut || {
                main.queue.submit(k.take());
                wait(main);
            }));
            let mut k = Some(k12(main, &km, &b));
            r[3].push(time(&mut || {
                main.queue.submit(k.take());
                wait(main);
            }));
            let (mut k, mut c) = (Some(k3(main, &km, &a)), Some(copy(t, &src, &dst)));
            r[4].push(time(&mut || {
                main.queue.submit(k.take());
                t.queue.submit(c.take());
                wait(main);
                wait(t);
            }));
            let (mut k, mut c) = (Some(k12(main, &km, &b)), Some(copy(t, &src, &dst)));
            r[5].push(time(&mut || {
                main.queue.submit(k.take());
                t.queue.submit(c.take());
                wait(main);
                wait(t);
            }));
            let (mut k, mut c) = (Some(k12(main, &km, &b)), Some(copy(main, &msrc, &mdst)));
            r[6].push(time(&mut || {
                main.queue.submit(k.take());
                main.queue.submit(c.take());
                wait(main);
            }));
        }
        let m: Vec<f64> = r.into_iter().map(median).collect();
        eprintln!("n = {n}, readback {} MB; transfer family {fam}", size / 1_000_000);
        eprintln!("copy alone: transfer queue {:.2} ms, main queue {:.2} ms", m[0], m[1]);
        eprintln!("K3 alone {:.2}, K1+K2 alone {:.2}", m[2], m[3]);
        eprintln!("K3 + copy (transfer queue): {:.2} ms (K3 alone {:.2}, sum {:.2})", m[4], m[2], m[2] + m[0]);
        eprintln!("K1+K2 + copy (transfer queue): {:.2} ms (K1+K2 alone {:.2}, sum {:.2})", m[5], m[3], m[3] + m[0]);
        eprintln!("K1+K2 then copy on main: {:.2} ms", m[6]);
    }
}
