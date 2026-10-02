//! The `Compressor` API against the CPU oracle: every frame must equal
//! `compress_block_to_frame` of its block, byte for byte, and decode to the block's bytes.
use std::sync::Arc;

use gzc_core::config::BLOCK_SIZE;
use gzc_core::frame::FrameOptions;
use gzc_core::params::{LVL3, MatchParams};
use gzc_core::reference::compress_block_to_frame;
use gzc_core::synth;
use gzc_gpu::pipeline::{PipelineConfig, max_batch_for_budget, vram_bytes_with};
use gzc_gpu::testing::{gpu, gpu_test_slot, gpu_with};
use gzc_gpu::{Compressor, CompressorOptions, Error, Frames, GpuContext, GpuOptions, GpuParams, Level};

const OPTS: FrameOptions = FrameOptions { checksum: false, huffman: true };

/// `len` bytes that mix text, texture-like data, a run of zeros and noise, so that the blocks
/// take the compressed, the RLE and the raw path.
fn sample(len: usize) -> Vec<u8> {
    let mut src = synth::text(1, 2 * BLOCK_SIZE + 123);
    src.extend(synth::zeros(BLOCK_SIZE + 999));
    src.extend(synth::dds_like(3, 2 * BLOCK_SIZE));
    src.extend(synth::random(7, BLOCK_SIZE / 2));
    src.extend(synth::nif_like(5, BLOCK_SIZE));
    src.iter().copied().cycle().take(len).collect()
}

/// Options for tests: the environment's GPU options and small batches, so a few blocks already
/// span several batches.
fn small(level: Level) -> CompressorOptions {
    CompressorOptions { batch_blocks: Some(3), inflight: 2, ..CompressorOptions::from_env(level) }
}

/// Each frame equals the oracle's frame of its block, declares the block's length, and the
/// frames together decode to the blocks.
fn check(frames: &Frames, blocks: &[&[u8]], preset: MatchParams, what: &str) {
    assert_eq!(frames.len(), blocks.len(), "{what}: frame count");
    for (i, (frame, block)) in frames.iter().zip(blocks).enumerate() {
        assert!(frame == compress_block_to_frame(block, preset, OPTS), "{what}: frame {i} differs from the CPU oracle");
        assert_eq!(frames.frame(i), frame, "{what}: frame({i})");
        let declared = zstd::zstd_safe::get_frame_content_size(frame).unwrap();
        assert_eq!(declared, Some(block.len() as u64), "{what}: content size of frame {i}");
    }
    if blocks.is_empty() {
        assert!(frames.as_bytes().is_empty(), "{what}: bytes without frames");
        return;
    }
    let all: Vec<u8> = blocks.concat();
    assert!(zstd::stream::decode_all(frames.as_bytes()).unwrap() == all, "{what}: the frames do not decode to the input");
}

/// `Compressor::compress` on odd lengths, for all four levels.
#[test]
fn compress_matches_cpu_oracle_every_level() {
    let _gpu = gpu_test_slot();
    let ctx = gpu();
    let lens = [0, 1, 255, 256, 65535, 65536, 65537, 3 * 65536 + 17, 7 * 65536 + 4097];
    for level in Level::ALL {
        let compressor = Compressor::with_context(ctx.clone(), &small(level)).unwrap();
        assert_eq!((compressor.batch_blocks(), compressor.preset()), (3, level.preset()));
        for len in lens {
            let data = sample(len);
            let frames = compressor.compress(&data).unwrap();
            let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();
            assert_eq!(frames.len(), len.div_ceil(BLOCK_SIZE));
            check(&frames, &blocks, level.preset(), &format!("{level:?}, {len} bytes"));
        }
    }
}

/// With no explicit batch the compressor takes the largest batch that fits the VRAM budget.
#[test]
fn batch_is_sized_by_the_vram_budget() {
    let _gpu = gpu_test_slot();
    let ctx = gpu();
    for level in [Level::Zstd9, Level::Zstd16] {
        let options = CompressorOptions { vram_budget_mib: 256, ..CompressorOptions::from_env(level) };
        let compressor = Compressor::with_context(ctx.clone(), &options).unwrap();
        let params = GpuParams { matching: level.preset(), emit_frames: true, huffman: true };
        let batch = compressor.batch_blocks() as u32;
        assert_eq!(batch, max_batch_for_budget(&ctx, params, 3, 256).unwrap());
        let mib = |batch| vram_bytes_with(&PipelineConfig { batch, inflight: 3, params }, ctx.direct_upload()).div_ceil(1 << 20);
        assert!(batch > 8 && mib(batch) <= 256 && mib(batch + 1) > 256, "{level:?}: batch {batch}, {} MiB", mib(batch));
        // More than 64 blocks per batch: several threads copy the frames out.
        let data = sample(200 * BLOCK_SIZE + 77);
        let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();
        check(&compressor.compress(&data).unwrap(), &blocks, level.preset(), &format!("{level:?}, budget"));
    }
}

/// `compress_blocks` takes short blocks anywhere, and rejects empty and oversized ones.
#[test]
fn compress_blocks_allows_short_blocks_anywhere() {
    let _gpu = gpu_test_slot();
    let ctx = gpu();
    let data = sample(8 * BLOCK_SIZE);
    let lens = [BLOCK_SIZE, 1, 300, BLOCK_SIZE, 4096, 65535, 2, 255, 256];
    let blocks: Vec<&[u8]> = lens.iter().enumerate().map(|(i, &len)| &data[i * 1000..i * 1000 + len]).collect();
    for level in [Level::Zstd3, Level::Zstd9, Level::Zstd16] {
        let compressor = Compressor::with_context(ctx.clone(), &small(level)).unwrap();
        check(&compressor.compress_blocks(&blocks).unwrap(), &blocks, level.preset(), &format!("{level:?}, short blocks"));
        assert!(compressor.compress_blocks(&[]).unwrap().is_empty());
        for bad in [&data[..0], &data[..BLOCK_SIZE + 1]] {
            let e = compressor.compress_blocks(&[&data[..10], bad]).unwrap_err();
            assert!(matches!(&e, Error::InvalidInput(m) if m.contains("block 1")), "{e:?}");
        }
        // The compressor still works after a rejected call.
        check(&compressor.compress_blocks(&blocks[..2]).unwrap(), &blocks[..2], level.preset(), "after an error");
    }
}

/// The streaming form: payloads written straight into the batches, some from other threads,
/// and the frames taken in place from each finished batch.
#[test]
fn stream_matches_cpu_oracle() {
    let _gpu = gpu_test_slot();
    let ctx = gpu();
    let payloads: Vec<Vec<u8>> =
        [100_000, 5_000, 65536, 1, 3 * 65536 + 17, 255, 70_000].iter().map(|&len| sample(len)).collect();
    let blocks: Vec<&[u8]> = payloads.iter().flat_map(|p| p.chunks(BLOCK_SIZE)).collect();
    // Which payloads go into which batch of 4 blocks: 2 + 1 + 1, then 1, then 4, then 1 + 2.
    let batches: [&[usize]; 4] = [&[0, 1, 2], &[3], &[4], &[5, 6]];
    for level in Level::ALL {
        let options = CompressorOptions { batch_blocks: Some(4), ..small(level) };
        let compressor = Compressor::with_context(ctx.clone(), &options).unwrap();
        let mut got: Vec<Option<Vec<u8>>> = vec![None; blocks.len()];
        let mut tags = Vec::new();
        let stats = compressor
            .stream(
                |batch| {
                    tags.push((batch.tag(), batch.first_index(), batch.len()));
                    for (index, frame) in batch.frames() {
                        assert!(got[index].replace(frame.to_vec()).is_none(), "block {index} delivered twice");
                    }
                    Ok(())
                },
                |stream| {
                    assert_eq!((stream.batch_blocks(), stream.submitted_blocks()), (4, 0));
                    for (tag, members) in batches.iter().enumerate() {
                        let mut batch = stream.next_batch()?;
                        assert_eq!((batch.capacity(), batch.len(), batch.is_empty()), (4, 0, true));
                        let lens: Vec<usize> = members.iter().map(|&p| payloads[p].len()).collect();
                        let mut regions = batch.reserve(&lens)?;
                        assert!(regions.iter().zip(&lens).all(|(r, &len)| r.len() == len && !r.is_empty()));
                        // The payloads are disjoint: fill each on its own thread.
                        std::thread::scope(|s| {
                            for (region, &p) in regions.iter_mut().zip(members.iter()) {
                                let payload = &payloads[p];
                                s.spawn(move || {
                                    for (k, chunk) in payload.chunks(1000).enumerate() {
                                        region.write(k * 1000, chunk);
                                    }
                                });
                            }
                        });
                        drop(regions);
                        let blocks_in: usize = lens.iter().map(|len| len.div_ceil(BLOCK_SIZE)).sum();
                        assert_eq!((batch.len(), batch.remaining()), (blocks_in, 4 - blocks_in));
                        let first = stream_first(&batches, &payloads, tag);
                        assert_eq!(batch.submit_tagged(tag as u64)?, first);
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(stats.batches, 4);
        assert_eq!(tags, [(0, 0, 4), (1, 4, 1), (2, 5, 4), (3, 9, 3)]);
        for (i, (frame, block)) in got.iter().zip(&blocks).enumerate() {
            let frame = frame.as_ref().unwrap_or_else(|| panic!("{level:?}: block {i} was not delivered"));
            assert!(*frame == compress_block_to_frame(block, level.preset(), OPTS), "{level:?}: frame {i} differs");
        }
    }
}

/// The index of the first block of batch `tag`.
fn stream_first(batches: &[&[usize]], payloads: &[Vec<u8>], tag: usize) -> usize {
    batches[..tag].iter().flat_map(|m| m.iter()).map(|&p| payloads[p].len().div_ceil(BLOCK_SIZE)).sum()
}

/// A batch never holds a block without its real length: lengths come with `reserve`, and bad
/// reservations and empty submissions are errors. The stream also takes whole blocks by copy.
#[test]
fn stream_rejects_bad_batches_and_passes_errors_through() {
    let _gpu = gpu_test_slot();
    let ctx = gpu();
    let compressor = Compressor::with_context(ctx, &small(Level::Zstd3)).unwrap();
    let data = sample(5 * BLOCK_SIZE + 9);
    let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();

    let mut got = Vec::new();
    compressor
        .stream(
            |batch| {
                got.extend(batch.frames().map(|(_, f)| f.to_vec()));
                Ok(())
            },
            |stream| {
                let mut batch = stream.next_batch()?;
                for bad in [&[0usize][..], &[10, 0], &[3 * BLOCK_SIZE + 1], &[BLOCK_SIZE, 2 * BLOCK_SIZE + 1]] {
                    let e = batch.reserve(bad).err().expect("a bad reservation");
                    assert!(matches!(e, Error::InvalidInput(_)), "{bad:?}: {e:?}");
                    assert!(batch.is_empty(), "{bad:?} reserved something");
                }
                let e = batch.submit().unwrap_err();
                assert!(matches!(&e, Error::InvalidInput(m) if m.contains("empty")), "{e:?}");
                // The batch was not submitted: the stream hands it out again.
                assert_eq!(stream.submitted_blocks(), 0);
                stream.compress_blocks(&blocks)?;
                assert_eq!(stream.submitted_blocks(), blocks.len());
                Ok(())
            },
        )
        .unwrap();
    for (i, (frame, block)) in got.iter().zip(&blocks).enumerate() {
        assert!(*frame == compress_block_to_frame(block, LVL3, OPTS), "frame {i} differs");
    }
    assert_eq!(got.len(), blocks.len());

    // An error from either closure comes back as it was, and the compressor stays usable.
    let e = compressor
        .stream(|_batch| Err(Error::InvalidInput("from on_batch".to_string())), |stream| stream.compress_blocks(&blocks))
        .unwrap_err();
    assert!(matches!(&e, Error::InvalidInput(m) if m == "from on_batch"), "{e:?}");
    let io = || Error::Other(Box::new(std::io::Error::other("from produce")));
    let e = compressor.stream(|_batch| Ok(()), |_stream| Err(io())).unwrap_err();
    assert!(matches!(&e, Error::Other(inner) if inner.to_string() == "from produce"), "{e:?}");
    check(&compressor.compress(&data).unwrap(), &blocks, LVL3, "after failed streams");
}

/// Bad options are typed errors, not panics.
#[test]
fn bad_options_are_typed_errors() {
    let _gpu = gpu_test_slot();
    let ctx = gpu();
    let build = |options: CompressorOptions| Compressor::with_context(ctx.clone(), &options).err().expect("an error");
    let base = || small(Level::Zstd3);
    // A lazy parse over the whole block: valid parameters the GPU does not implement.
    let e = build(CompressorOptions { preset: gzc_core::fixtures::LVL9, ..base() });
    assert!(matches!(&e, Error::Unsupported(m) if m.contains("segment")), "{e:?}");
    let e = build(CompressorOptions { preset: MatchParams { depth: 0, ..LVL3 }, ..base() });
    assert!(matches!(&e, Error::InvalidInput(m) if m.contains("depth")), "{e:?}");
    let e = build(CompressorOptions { inflight: 0, ..base() });
    assert!(matches!(&e, Error::InvalidInput(m) if m.contains("inflight")), "{e:?}");
    let e = build(CompressorOptions { batch_blocks: Some(0), ..base() });
    assert!(matches!(&e, Error::InvalidInput(m) if m.contains("batch")), "{e:?}");
    let e = build(CompressorOptions { batch_blocks: None, vram_budget_mib: 1, ..base() });
    assert!(matches!(&e, Error::InvalidInput(m) if m.contains("1 MiB")), "{e:?}");
}

/// A lost device is `Error::DeviceLost`, from a running compressor and from building one.
#[test]
fn a_lost_device_is_a_typed_error() {
    let _gpu = gpu_test_slot();
    // Its own context: losing the device must not affect other tests.
    let options = GpuOptions { direct_upload: Some(false), transfer_queue: false, ..GpuOptions::from_env() };
    let ctx: Arc<GpuContext> = gpu_with(options);
    let compressor = Compressor::with_context(ctx.clone(), &small(Level::Zstd3)).unwrap();
    let data = sample(2 * BLOCK_SIZE);
    assert_eq!(compressor.compress(&data).unwrap().len(), 2);
    ctx.device().destroy();
    let _ = ctx.device().poll(wgpu::PollType::Poll);
    assert!(ctx.device_lost().is_some(), "the device-lost callback did not fire");
    let e = compressor.compress(&data).unwrap_err();
    assert!(matches!(e, Error::DeviceLost(_)), "{e:?}");
    drop(compressor);
    let e = Compressor::with_context(ctx, &small(Level::Zstd3)).err().expect("a compressor on a lost device");
    assert!(matches!(e, Error::DeviceLost(_)), "{e:?}");
}
