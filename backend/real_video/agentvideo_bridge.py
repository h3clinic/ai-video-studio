"""Load an explicit pinned upstream checkout, without running its rendering route."""
import hashlib
import subprocess
import sys
from pathlib import Path

COMMIT='5649783d68f9d424fde94e66bfa4af8f1754f2f8'


def load_upstream(path):
    root=Path(path).resolve()
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip()
    if head!=COMMIT:raise ValueError('Unreviewed AgentVideo revision')
    # No checkout mutation, dependency installer, image crawler, or Metal import.
    for name in ('agentvideo/__init__.py','agentvideo/agents/__init__.py','agentvideo/base.py',
                 'agentvideo/bus.py','agentvideo/llm.py','agentvideo/agents/decision.py'):
        dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--',name], text=True)
        if dirty:raise ValueError('Modified upstream core: '+name)
    existing=sys.modules.get('agentvideo')
    if existing and Path(existing.__file__).resolve()!=root/'agentvideo/__init__.py':
        raise ValueError('Another AgentVideo package is already loaded')
    sys.path.insert(0,str(root))
    from agentvideo.base import Agent, AgentFailed
    from agentvideo.bus import Bus
    from agentvideo.agents.decision import IdeaModel, DecisionAgent, script_issues
    return dict(Agent=Agent, AgentFailed=AgentFailed, Bus=Bus, IdeaModel=IdeaModel,
                DecisionAgent=DecisionAgent,script_issues=script_issues,
                provenance=dict(commit=head, core={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                    for name in ('agentvideo/base.py','agentvideo/bus.py','agentvideo/llm.py','agentvideo/agents/decision.py')}))
