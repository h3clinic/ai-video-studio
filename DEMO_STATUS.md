# Gaussian Studio integration

## Latest sponsor-demo build — 2026-10-04

- A real 17-worker run completed, reusing 14 existing briefs and making three new
  ElevenLabs sound-generation requests. Those briefs are not verified geometry.
- The app exported a five-second MP4 containing three independently adjustable
  sound layers. The export copies the source video stream, makes zero new model
  API calls and leaves visuals unchanged. All 121 decoded RGB frames match the
  original. Mixing/muxing measured 0.265 seconds for this single run; this is not
  model-generation time or a claim of Gaussian/Wan speedup.
- Current runs are prominent; earlier failures remain available in history.
  Agent Observatory can filter the newest run/team or show the full history.
- The SpacetimeDB 2.10.2 runtime-codegen/CSP conflict is fixed using interpreted
  codecs. A real local handshake and six scoped subscriptions passed with string
  code generation disabled. No unsafe-eval exception was added.
- 37 Electron tests and the production build passed. Native app verification and
  a fresh 14-check real local database integration suite passed. The native app
  connected and committed revision 1 of a real instruction with attributed time.
  The final edited demo is being completed separately; tests are not video QA.
- Contents distinguishes audio, workflow and geometry parts. A sound task with a
  verified audio artifact is no longer mislabeled as missing Gaussian geometry.
- 405 backend Python sources and a SHA-256 manifest are included. Models, provider
  keys, source media, AgentVideo, Python and the database runtime are not bundled.

Gaussian fruit replacement, exact mouth contact, verified part bindings, internet
collaboration and a signed standalone installer remain unfinished. Do not call this
a production-ready autonomous Gaussian video generator.

## Archived implementation log

Everything below records earlier checkpoints. Counts, credential errors and
"current" labels in this archive are historical, not the latest app status.

## Current correction — 2026-10-04

Latest repair session: the existing RunPod key is now left enabled permanently,
at the user's explicit request, with unchanged scopes. It is never exposed to
the renderer. All experiment Pods were verified EXITED and shutdown credentials
were removed from their environments; permanent access is not ongoing spending
authorization. No top-ups were made.

Authentication is restored. Jupyter's workspace-listing failure is isolated:
authenticated file/metadata reads succeed but listing can return 404. The bounded
runner now creates a new experiment directory exclusively through its authenticated
kernel, then verifies uploaded chunks. Trial v6 reached remote model execution.

The empty-mask placement crash is fixed. Trial v6 completed a five-frame diagnostic
on RTX 4090, but made ZERO replacements: conflicting whole-fruit/slice/peel masks
protect almost every target pixel. Sampled frames 0, 2 and 4 were visually rejected
as an apple edit. It is not a new five-second video, changed generator weights,
or measured compute saving. Exact evidence is in the prototype's
artifacts/cloud/apple-observed-parts-20261004-v6. The rejected static worker remains
disabled in this UI; successful unit tests do not unblock it.

Current checks: 122 focused Python tests plus three runner regression tests and
18 Electron boundary/UI tests pass. Production frontend build succeeds. General
ownership arbitration, fitted apple-slice/peel/bite assets, contact consistency,
and a signed standalone deployment remain unfinished. This is not production-ready.

The latest desktop retest also exposed an Undici assertion during state polling.
All main-process loopback JSON requests now use Electron net.fetch, not just
video streams, and startup consumes the bounded response body before readiness.
The app was cleanly restarted; the original session UI displayed the v6 rejected
diagnostic and preserved earlier failures. No model evaluation ran on the laptop.

The following notes preserve earlier steps; HTTP401/key-disabled statements below
describe historical failures, not current access.

Latest: all 14 original GIF role cards are active, with typed per-agent/part
task submissions and owner-configured Gemini credentials (no end-user key box).
Canonical apple IDs are checksum-bound to the existing 20,000-point asset;
this does not validate scene contact or the rejected coarse geometry. Scoped
motion/recolour operators and a guarded remote worker are implemented. Texture
painting is blocked until real UV correspondence exists.

One live Gemini Vector Agent call returned valid five-second turn-and-return
controls and a 15% red recolour. GPU execution is blocked by RunPod HTTP401;
no new animated video was produced. 47 Python tests and 9 Electron boundary
tests pass; production build succeeds. These do not establish video quality.

Desktop verification via the computer-use skill confirmed the restored UI,
preserved task history and blocked Vector Agent result. Opening media exposed
an Undici stream assertion; the media proxy now uses Electron net.fetch.
Retest loaded the real historical MP4 and advanced playback to 5.04 seconds
without that error. The old screenshot is still historical, not this build.

The original App.tsx layout is ACTIVE again. Desktop chat uses real backend
jobs, with persistent projects, renaming and recoverable archiving. Gemini
planning runs through AgentVideo; proposed part owners remain explicitly
unbound to Gaussian IDs. Generic reference prompts are project-scoped.

The new remote apple experiment completed but was REJECTED: floating apple,
blurry bowl, missing slices/peel/contact. It edits recorded motion; no new
weights or Gemini-asset 3D fitting were used. Pod EXITED; temporary key disabled.

Current build and eight Electron boundary tests pass. Staging now includes
380 Python sources. The older smoke screenshot and statements below describe
the previous dashboard, NOT verification of this restored UI. Renewed visual
verification remains outstanding. General part fitting/edit execution and a
signed standalone installer remain unfinished; this is not production-ready.

## Historical dashboard notes (superseded where stated above)

## Run on this workstation

Run `launch-demo.ps1` after building. The locked package manager is pnpm;
`pnpm install`, `node node_modules/electron/install.js`, then `pnpm build`.
The legacy npm lockfile was replaced by pnpm-lock.yaml and remains recoverable in Git.

The Electron process owns a token-authenticated Python service on loopback port
8790. Set GAUSSIAN_PROJECT_ROOT and GAUSSIAN_PYTHON to override the development
paths. No model inference runs on the laptop. Credentials use Windows DPAPI,
outside the repository; there is no API key read-back endpoint.

## Verified

- TypeScript check and Vite production build.
- Six Electron boundary tests; twenty backend/credential tests.
- Electron smoke test with real media and seven ownership/protection entries.
- No paid calls during the app smoke test.
- Runtime dependency audit: zero advisories at this check (not a security guarantee).
- Source staging copies 376 Python modules and a SHA-256 manifest, excluding
  secrets, environments, model weights and experimental datasets.

## Actual functionality

- Gemini image-reference generation, with persisted job status and no retry.
- Gemini, Runway and RunPod encrypted credential storage in the backend.
- Existing RunPod connection/status, bounded remote CUDA preflight and stop request.
- Explicit part-agent assignments and blocked/unverified ownership display.
- Real image/video previews; rejected historical output stays labeled rejected.
- Interrupted jobs are not silently restarted after an app restart.

## Not production-ready yet

- Remote fitting, per-part edit execution and video rendering are not wired as
  submitted jobs. Connecting a Pod does not implement those stages.
- Part IDs are not verified material correspondences; Gaussian counts remain
  unknown. The UI does not fabricate successful agents or edits.
- No automatic Pod provisioning or budget watchdog has been exposed in this UI.
- Runway generation is not exposed in this build.
- Source packaging is configured, but a self-contained signed installer and
  bundled minimal Python runtime are not built or validated.
- The legacy App.tsx demo/auth views remain source references but are excluded
  from the active entry point. Their canned chat is not shown as live work.
- This is a local, single-user demo, not a multi-user deployment.

Do not present this as an autonomous Gaussian-video generation demo yet.
