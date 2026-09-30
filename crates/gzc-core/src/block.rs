//! File-to-block chunking with zero-padding of the final block.
use crate::config::BLOCK_SIZE;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub data: Vec<u8>,
    pub real_len: usize,
}

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
