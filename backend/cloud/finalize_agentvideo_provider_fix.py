"""Export already-recorded evidence without making another API call."""
import json
from real_video.gaussian_program import ROOT
from real_video.studio_agent_events import project_trace

base=ROOT/'artifacts/cloud/agentvideo_gemini_budget_fix_live_v1'
trace=json.loads((base/'trace.json').read_text())
(base/'studio_events.json').write_text(json.dumps(project_trace(trace),indent=2),encoding='utf-8')
result=json.loads((base/'result.json').read_text())
review=dict(status='live_director_verified',
    observation='Original IdeaModel script verifier passed with Gemini transport after provider allocation floor rose from requested150 to2048.',
    mechanism='Successful call used859 reasoning tokens and80 answer tokens. This supports inadequate token headroom as the integration defect; the earlier incomplete-response finish reason was not captured.',
    actual=result['usage'],
    regressions='Incomplete responses still rejected; full allocated tokens reserved; missing candidate and transport failures recorded.',
    remaining=['Full specialist end-to-end integration','Remote renderer port','React studio event subscription','Verified Gaussian edit masks and execution'],
    video_generated=False,weights_modified=False)
(base/'fix_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
print(json.dumps(review,indent=2))
