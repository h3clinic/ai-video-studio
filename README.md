# AI Video Studio

A Windows Electron research application for observable agent jobs, project history,
private local SpacetimeDB collaboration, generated sound layers and local audio mixing.
The checked-in Python backend is source code, not a bundled model or turnkey video generator.
The full orange-to-apple Gaussian scene replacement remains rejected and disabled.

## Run a source checkout

Requirements: Windows, Node.js with pnpm, Python 3.11 or 3.12, and Git.

```powershell
pnpm install --frozen-lockfile
node node_modules/electron/install.js
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "requests>=2.32,<3" "imageio-ffmpeg>=0.6,<0.7" "numpy>=2,<3" "Pillow>=11,<13"
$env:GAUSSIAN_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
pnpm build
.\launch-demo.ps1
```

The default build verifies `backend/source-manifest.json` against the included Python
files. It needs no sibling research checkout. The listed Python dependencies support
startup, provider orchestration and audio mixing; optional Gaussian operations require
additional research dependencies and separately obtained assets. No model installation
or paid compute starts during build or application launch.

Development preserves the original workspace integration: if
`../../outputs/gaussian_vector_prototype` exists relative to this repository, it is used
with the existing sibling Python environment. Otherwise the app uses its own `backend/`
directory. Set `GAUSSIAN_PROJECT_ROOT` and `GAUSSIAN_PYTHON` to absolute paths to select a
different full backend project and Python executable. Jobs and generated media live
under that project's `artifacts/`, which Git ignores.

## Optional agent planning and collaboration

Gemini agent planning requires the separately obtained
[AgentVideo checkout](https://github.com/h3clinic/agentvideo) at commit
`5649783d68f9d424fde94e66bfa4af8f1754f2f8`, with its core files unchanged. It is not
redistributed here. The backend expects it at `PROJECT_ROOT/../../work/agentvideo-reference`.
For a standalone source checkout using `backend/`, that resolves to
`../work/agentvideo-reference` relative to this repository. Review its usage terms and
check out the pinned revision yourself; the application neither clones it nor installs
its local model dependencies. The Gemini adapter uses the upstream agent protocol without
running the Apple-specific rendering pipeline.

ElevenLabs credentials can be entered in Environment → Connect ElevenLabs. Gemini and
Runway currently use the separate loopback settings form: from the backend project,
run `python -m real_video.runway_settings --provider gemini --port 8780` (or
`--provider runway --port 8779`) and open the printed local URL. RunPod setup remains
a separate owner configuration step; launch does not provision a GPU automatically.
Credentials are encrypted with Windows DPAPI outside the repository, never embedded
in the frontend or distributed with the backend. Provider features require your own
access and may incur charges. No personal API key is included in a public checkout.

The collaboration module is in `spacetimedb/collaboration/`, with generated client bindings
in `src/module_bindings/collaboration/`. It targets SpacetimeDB 2.10.2, database
`ai-video-studio-collaboration`, on `127.0.0.1:3000`. Install, start and publish that local
module separately. `integration.test.ts` exercises real server identities, membership,
revision conflicts, edit history and revocation; it is not an in-memory database mock.
An unavailable database is shown as disconnected and does not disable local jobs.
The SDK is pinned to 2.10.2. `src/spacetimeCsp.ts` provides interpreted product/sum
codecs because that SDK otherwise uses runtime compilation, forbidden by our strict
Content Security Policy. Run `node --disallow-code-generation-from-strings
tools/check-collaboration-csp.cjs` against the running local module to verify the
handshake and six subscriptions without relaxing CSP. This check performs no edits.

## Packaging and source maintenance

`pnpm package` includes the verified Python source under the installation's
`resources/backend`. On launch, the packaged application materializes only manifest-listed
Python files in the writable Electron user-data directory under
`outputs/gaussian_vector_prototype`; artifacts remain there across launches. Files changed
manually in that source mirror are not silently overwritten. Set `GAUSSIAN_PROJECT_ROOT`
to override this writable project and `GAUSSIAN_PYTHON` to an existing compatible Python
executable. A Python runtime, model weights, source videos, generated effects, credentials,
the AgentVideo checkout and SpacetimeDB are not included. A signed standalone installer
has not been validated.

To refresh the vendored backend intentionally from a reviewed research checkout:

```powershell
pnpm backend:refresh C:/path/to/gaussian_vector_prototype
pnpm backend:verify
pnpm test
pnpm build
```

Refresh copies only top-level Python files in `real_video`, `cloud`, `gv` and `tests`
and regenerates their SHA-256 manifest. Inspect the diff before publication. It does not
copy datasets, secrets, runtime history, model weights or experiment media. Existing
unlisted Python files cause verification to fail and require manual review.

The original repository's artwork remains attributed by its Git history. No new license
grant is asserted for upstream code, artwork or model assets. See [DEMO_STATUS.md](DEMO_STATUS.md)
for research limitations; historical measurements there are not fresh release validation.
