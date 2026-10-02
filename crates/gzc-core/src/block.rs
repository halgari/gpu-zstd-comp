//! File-to-block chunking with zero-padding of the final block.
use crate::config::BLOCK_SIZE;

/// One block of a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The block's bytes, zero-padded to `BLOCK_SIZE`.
    pub data: Vec<u8>,
    /// How many bytes of `data` are the file's: `BLOCK_SIZE`, or less for a file's last block.
    pub real_len: usize,
}

impl Block {
    /// The block's real bytes, what its frame holds (`reference::compress_block_to_frame`).
    pub fn real(&self) -> &[u8] {
        &self.data[..self.real_len]
    }
}

/// Cuts `bytes` into `BLOCK_SIZE` blocks, zero-padding the last one. Empty input gives no
/// blocks.
pub fn chunk_file(bytes: &[u8]) -> Vec<Block> {
    bytes
        .chunks(BLOCK_SIZE)
        .map(|c| {
            let mut data = vec![0u8; BLOCK_SIZE];
            data[..c.len()].copy_from_slice(c);
            Block { data, real_len: c.len() }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BLOCK_SIZE;

    #[test]
    fn empty_file_has_no_blocks() {
        assert!(chunk_file(&[]).is_empty());
    }

    #[test]
    fn last_block_is_zero_padded() {
        let data = vec![7u8; BLOCK_SIZE + 1];
        let b = chunk_file(&data);
        assert_eq!(b.len(), 2);
        assert_eq!((b[0].real_len, b[1].real_len), (BLOCK_SIZE, 1));
        assert_eq!(b[1].data.len(), BLOCK_SIZE);
        assert_eq!(b[1].data[0], 7);
        assert!(b[1].data[1..].iter().all(|&x| x == 0));
    }
}
