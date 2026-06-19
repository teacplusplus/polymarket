use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

pub const XFRAMES_DIR_ENV: &str = "XFRAMES_DIR";
pub const XFRAMES_LAST_DAYS_ENV: &str = "XFRAMES_LAST_DAYS";
pub const DEFAULT_XFRAMES_DIR: &str = "xframes";
pub const DEFAULT_XFRAMES_LAST_DAYS: usize = 3;

static XFRAMES_DATE_DIR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{4}-\d{2}-\d{2}$").expect("valid xframes date-dir regex")
});

pub fn xframes_root() -> PathBuf {
    std::env::var(XFRAMES_DIR_ENV)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_XFRAMES_DIR))
}

pub fn xframes_path(path: impl AsRef<Path>) -> PathBuf {
    xframes_root().join(path)
}

pub fn xframes_last_days() -> usize {
    std::env::var(XFRAMES_LAST_DAYS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&days| days > 0)
        .unwrap_or(DEFAULT_XFRAMES_LAST_DAYS)
}

pub fn last_xframes_date_dirs(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut date_paths: Vec<PathBuf> = paths
        .into_iter()
        .filter(|path| path.is_dir() && is_xframes_date_dir(path))
        .collect();
    date_paths.sort();

    let last_days = xframes_last_days();
    if date_paths.len() > last_days {
        date_paths.split_off(date_paths.len() - last_days)
    } else {
        date_paths
    }
}

pub fn is_xframes_date_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|name| XFRAMES_DATE_DIR_RE.is_match(name))
}

pub fn graph_root() -> PathBuf {
    sibling_root("graph")
}

pub fn streams_root() -> PathBuf {
    sibling_root("streams")
}

pub fn graph_html_path_from_xframes_bin(bin_path: &Path) -> Option<PathBuf> {
    if let Some(rel) = strip_configured_xframes_root(bin_path) {
        return Some(graph_root().join(rel).with_extension("html"));
    }
    graph_html_path_from_legacy_xframes_segment(bin_path)
}

fn sibling_root(name: &str) -> PathBuf {
    let root = xframes_root();
    root.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.join(name))
        .unwrap_or_else(|| PathBuf::from(name))
}

fn strip_configured_xframes_root(path: &Path) -> Option<PathBuf> {
    let root = xframes_root();
    if let Ok(rel) = path.strip_prefix(&root) {
        return Some(rel.to_path_buf());
    }

    let cwd = std::env::current_dir().ok()?;
    if root.is_relative() {
        let abs_root = cwd.join(&root);
        if let Ok(rel) = path.strip_prefix(abs_root) {
            return Some(rel.to_path_buf());
        }
    } else if path.is_relative() {
        let abs_path = cwd.join(path);
        if let Ok(rel) = abs_path.strip_prefix(&root) {
            return Some(rel.to_path_buf());
        }
    }

    None
}

fn graph_html_path_from_legacy_xframes_segment(bin_path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    let mut switched = false;
    for comp in bin_path.components() {
        if comp.as_os_str() == DEFAULT_XFRAMES_DIR {
            out.push("graph");
            switched = true;
        } else {
            out.push(comp);
        }
    }
    switched.then(|| out.with_extension("html"))
}
