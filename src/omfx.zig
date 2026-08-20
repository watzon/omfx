//! omfx fork identity. This file is fork-owned and does not exist upstream.
//! Keep fork constants here so shared upstream files only need one-line hooks.

pub const distribution = "omfx";
pub const upstream_repo = "vercel-labs/fx";
pub const fork_repo = "watzon/omfx";

/// The upstream CDN at fx.sh serves upstream fx binaries. An omfx build must
/// never replace itself with an upstream binary, so every code path that
/// downloads from that CDN checks this flag. Turn it on only after omfx has
/// its own release channel.
pub const upstream_upgrades_enabled = false;
