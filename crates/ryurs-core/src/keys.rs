use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

#[derive(Debug, Clone, Copy, Default)]
pub struct KeyInventory {
    pub prod_count: usize,
    pub title_count: usize,
}

impl KeyInventory {
    pub fn total_count(self) -> usize {
        self.prod_count + self.title_count
    }
}

#[derive(Debug, Clone, Default)]
pub struct KeySet {
    pub prod: BTreeMap<String, Vec<u8>>,
    pub title: BTreeMap<String, Vec<u8>>,
}

impl KeySet {
    pub fn read_from_paths(prod_path: Option<&Path>, title_path: Option<&Path>) -> Result<Self> {
        let prod = match prod_path {
            Some(path) => {
                let bytes = std::fs::read(path).with_context(|| {
                    format!("failed to read prod key file from {}", path.display())
                })?;
                parse_keys_file(&bytes, &path.display().to_string())?
            }
            None => BTreeMap::new(),
        };

        let title = match title_path {
            Some(path) => {
                let bytes = std::fs::read(path).with_context(|| {
                    format!("failed to read title key file from {}", path.display())
                })?;
                parse_keys_file(&bytes, &path.display().to_string())?
            }
            None => BTreeMap::new(),
        };

        Ok(Self { prod, title })
    }

    pub fn inventory(&self) -> KeyInventory {
        KeyInventory {
            prod_count: self.prod.len(),
            title_count: self.title.len(),
        }
    }
}

fn parse_keys_file(bytes: &[u8], source: &str) -> Result<BTreeMap<String, Vec<u8>>> {
    let text = std::str::from_utf8(bytes)
        .with_context(|| format!("key file {source} is not valid UTF-8 text"))?;
    let mut keys = BTreeMap::new();

    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let (name, value) = trimmed.split_once('=').ok_or_else(|| {
            anyhow!("invalid key entry at {source}:{line_number}: expected key=value")
        })?;

        let key_name = name.trim();
        if key_name.is_empty() {
            bail!("invalid key entry at {source}:{line_number}: missing key name");
        }

        let key_bytes = decode_hex(value.trim(), source, line_number)?;
        keys.insert(key_name.to_string(), key_bytes);
    }

    Ok(keys)
}

fn decode_hex(value: &str, source: &str, line_number: usize) -> Result<Vec<u8>> {
    let hex = value.strip_prefix("0x").unwrap_or(value);
    if hex.is_empty() {
        bail!("invalid hex value at {source}:{line_number}: value is empty");
    }
    if hex.len() % 2 != 0 {
        bail!(
            "invalid hex value at {source}:{line_number}: odd number of hex digits ({})",
            hex.len()
        );
    }

    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let raw = hex.as_bytes();
    for pair in raw.chunks(2) {
        let hi = decode_nibble(pair[0])
            .ok_or_else(|| anyhow!("invalid hex value at {source}:{line_number}: bad character"))?;
        let lo = decode_nibble(pair[1])
            .ok_or_else(|| anyhow!("invalid hex value at {source}:{line_number}: bad character"))?;
        bytes.push((hi << 4) | lo);
    }

    Ok(bytes)
}

fn decode_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{KeySet, parse_keys_file};

    #[test]
    fn parses_prod_and_title_inventories() {
        let prod = parse_keys_file(
            b"# comment\nheader_key = 00112233\nmaster_key_00=0xabcdef01\n",
            "prod.keys",
        )
        .expect("prod.keys should parse");
        let title =
            parse_keys_file(b"titlekek_00=0011\n", "title.keys").expect("title.keys should parse");

        let keyset = KeySet { prod, title };
        let inventory = keyset.inventory();
        assert_eq!(inventory.prod_count, 2);
        assert_eq!(inventory.title_count, 1);
        assert_eq!(inventory.total_count(), 3);
    }

    #[test]
    fn rejects_invalid_hex_length() {
        let err =
            parse_keys_file(b"bad=001\n", "prod.keys").expect_err("odd-length hex should fail");
        assert!(err.to_string().contains("odd number of hex digits"));
    }

    #[test]
    fn rejects_missing_separator() {
        let err = parse_keys_file(b"bad_line\n", "prod.keys")
            .expect_err("line without separator should fail");
        assert!(err.to_string().contains("expected key=value"));
    }
}
