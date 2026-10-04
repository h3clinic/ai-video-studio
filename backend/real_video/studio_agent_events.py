"""Project actual AgentVideo events for the studio; no timed fake progress.

A verifier pass is a decision verdict, never evidence that a video was generated.
"""
def project_trace(trace):
    agents=trace.get('agents',{})
    rows={name:dict(id=name,name=name,role=a.get('role'),parent=a.get('parent'),
                    activity='registered',passes=0,failures=0,fallbacks=0)
          for name,a in agents.items()}
    events=[]
    for index,event in enumerate(trace.get('events',[])):
        name=event.get('agent')
        if name not in rows:raise ValueError('Event refers to unregistered agent')
        kind=event.get('kind')
        if kind not in ('spawn','prompt','response','pass','fail','fallback','info'):
            raise ValueError('Unknown AgentVideo event')
        row=rows[name]
        if kind=='prompt':row['activity']='requesting_model'
        elif kind=='response':row['activity']='proposal_received'
        elif kind=='fail':row['activity']='decision_rejected';row['failures']+=1
        elif kind=='pass':row['activity']='decision_verified';row['passes']+=1
        elif kind=='fallback':row['activity']='fallback';row['fallbacks']+=1
        events.append(dict(sequence=index,agent=name,kind=kind,seconds=event.get('t'),message=event.get('msg')))
    return dict(agents=list(rows.values()),events=events,agent_count=len(rows),
                video_status='unverified',source='actual_agentvideo_bus',
                note='Trace does not establish artifact quality or generation completion.')
