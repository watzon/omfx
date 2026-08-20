# How to keep omfx current with upstream fx

omfx is a fork of [vercel-labs/fx](https://github.com/vercel-labs/fx). The fork adds batteries on top of fx while it keeps the upstream speed, size, and shell-native form factor. Upstream moves fast. This document is the contract that keeps merges cheap. Follow it for every change and every sync.

## Remotes

The repository uses two remotes:

```bash
git remote add origin git@github.com:watzon/omfx.git
git remote add upstream https://github.com/vercel-labs/fx.git
```

`main` is the omfx default branch. It carries the upstream history plus the omfx delta. Never force-push `main`.

## Divergence rules

The omfx delta must stay small, additive, and enumerable.

1. Put new capabilities in new files. A new tool goes in a new directory under `src/tools/`. A new module goes in a new file. Fork-owned files never conflict on merge.
2. When a shared upstream file needs a hook, add the smallest possible edit and mark it with an `// omfx:` comment on its own line or at the start of the block. To enumerate the full delta in shared files, run:

   ```bash
   grep -rn "omfx" src/ build.zig
   ```

3. Do not rename upstream identifiers, paths, or strings for branding. The config directory stays `~/.fx/`. Environment variables stay `FX_*`. The WASM and Node-API artifacts stay `fx-core`, `fx-term`, and `libfx`. The in-app help header stays upstream's. The binary name and the version suffix carry the omfx identity.
4. Do not fork upstream logic in place. If a feature needs different behavior, add a seam (flag, registry entry, new file) and keep the upstream code path intact.
5. If upstream ships a feature that omfx already has, adopt the upstream version and delete ours in the same merge.
6. If a change is a general fix rather than an omfx battery, send it upstream first. Carry it in omfx only while the upstream review is open.

## Fork-owned files

These files exist only in omfx. Upstream merges never touch them:

* `src/omfx.zig`: fork identity constants and the `upstream_upgrades_enabled` flag.
* `UPSTREAM.md`: this document.
* `src/core/providers/`: direct provider registry, credentials, router, and OAuth flows.
* `src/gateway/openai_json.zig`, `src/gateway/openai_stream_provider.zig`: OpenAI-compatible wire codecs and transport.
* `tests/e2e/direct-providers.test.ts`: deterministic coverage for direct provider routing.

Shared files with `// omfx:` hooks at the time of writing: `build.zig` (binary name and `fx` alias install), `src/main.zig` (version suffix), `src/core/upgrade/auto_upgrade.zig` (upgrade gate), `src/core/cli/cli_surface.zig` (upgrade gate, provider login/logout), `src/builtins/gateway.zig` (provider router), `src/builtins/commands.zig` (login/logout usage text), `src/core/cli/cli_ask.zig` and `src/core/app/app_auth_runtime.zig` (direct credentials satisfy the credential gate), and a fork notice at the top of `AGENTS.md`. The grep command above is authoritative; this list is a snapshot.

## Sync procedure

Sync weekly, and before you start any new feature branch.

1. Fetch upstream:

   ```bash
   git fetch upstream --tags
   ```

2. Merge on a branch, not directly on `main`:

   ```bash
   git checkout -b sync/upstream-$(date +%Y-%m-%d) main
   git merge upstream/main
   ```

3. Resolve conflicts with this policy:
   * In shared files, take the upstream side, then re-apply the `// omfx:` hooks.
   * If upstream modified a workflow that omfx deleted (see CI below), keep it deleted.
   * If upstream changed a seam an omfx feature depends on, adapt the omfx side, not the upstream side.

4. Prove the merge:

   ```bash
   zig fmt --check src/ build.zig
   zig build
   zig build test
   ./zig-out/bin/omfx --version
   cd tests/e2e && bun install && bun test cli.test.ts
   ```

5. Open a PR from the sync branch into `main` and let CI pass before merging.

## Versioning

`pub const version` in `src/main.zig` is `<upstream version>+omfx.<n>`. The numeric part always equals the upstream version the fork last merged. Bump `<n>` for omfx releases on the same upstream base. Reset `<n>` to 1 when the upstream base changes. The upstream version parser reads only the numeric part, so keep the numeric part first and unchanged.

## Upgrades are disabled

Upstream fx auto-upgrades from the fx.sh CDN, which serves upstream binaries. An omfx build that upgraded from that CDN would replace itself with upstream fx. `src/omfx.zig` therefore sets `upstream_upgrades_enabled = false`, which turns off the background auto-upgrade and makes `fx upgrade` fail with an explanation. Do not remove this gate until omfx has its own release channel, and then point the upgrade code at that channel instead.

## CI on the fork

omfx keeps the upstream workflows that run self-contained on GitHub-hosted runners and deletes the ones that publish to Vercel infrastructure.

| Workflow | Status | Reason |
| --- | --- | --- |
| `ci.yml` | kept | Self-contained build and test. |
| `full-ci.yml` | kept | Uses standard hosted runner labels. Watch the first runs; trim the matrix if a runner label is unavailable. |
| `binary-size.yml` | kept | Self-contained size report. |
| `bench.yml` | kept | The PR path is self-contained. The upload step on `main` needs `BLOB_READ_WRITE_TOKEN` and may fail; remove or guard it if it stays red. |
| `pgso-macos-arm64.yml` | kept | Release qualification. Informational on the fork. |
| `release.yml`, `dev-release.yml`, `prepare-release.yml`, `publish-libfx.yml`, `cdn-backfill.yml` | deleted | They publish to the Vercel CDN, npm, and deploy hooks with secrets the fork does not have. |

When omfx grows its own release pipeline, write new fork-owned workflows instead of resurrecting the deleted ones.

## License obligations

fx is Apache-2.0. The fork must keep `LICENSE` and `THIRD_PARTY_NOTICES.md`, keep upstream attribution in `README.md`, and state its changes. This document and the `// omfx:` markers are that statement of changes.
