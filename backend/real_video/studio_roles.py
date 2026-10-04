"""Original Studio role identities, backed by tasks rather than GIF progress."""
ROLES = (
    ('idea-model', 'Idea Model', 'Scene intent and script'),
    ('video-controller', 'Video Controller', 'Dependency ordering and bounded execution'),
    ('crawler', 'Crawler Agent', 'Reference requirements; no unrestricted browsing'),
    ('decision', 'Decision Agent', 'Choose measured material and geometry operations'),
    ('vision-sensor', 'Vision Sensor', 'Inspect source evidence; no invented observations'),
    ('section-leads', 'Section Leads', 'Coordinate appearance, texture and motion'),
    ('field-agents', 'Field Agents', 'Validate bounded appearance parameters'),
    ('surroundings-list', 'Surroundings List Agent', 'Inventory related and protected parts'),
    ('kind-checkers', 'Kind Checkers', 'Check object and detached-part semantics'),
    ('mini-object-agents', 'Mini Object Agents', 'Object placement and contact requirements'),
    ('task-assigner', 'Task Assigner', 'Assign existing part ownership, never invent IDs'),
    ('vector-agents', 'Vector Agents', 'Rigid Gaussian mean and covariance motion'),
    ('sound-agents', 'Sound Agents', 'Bounded ElevenLabs sound effects; synchronization requires review'),
    ('verifiers', 'Verifiers', 'Check invariants and reject unsupported quality claims'),
)
ROLE_IDS = frozenset(row[0] for row in ROLES)


def role_state(projects, jobs, parts):
    jobs=[row for job in jobs for row in ([job]+[dict(w,agentId=w['roleId'],projectId=job['projectId'],stage=w.get('summary',w['task'])) for w in job.get('workers',[])])]
    result = []
    for project in projects:
        pid = project['id']
        for index, (role, name, task) in enumerate(ROLES):
            matches = [j for j in jobs if j.get('projectId') == pid and j.get('agentId') == role]
            latest = matches[-1] if matches else None
            result.append(dict(id=role, roleId=role, name=name, gif=f'/gifs/agent-{49+index}.gif',
                model='ElevenLabs SFX (owner credential)' if role=='sound-agents' else 'Gemini (owner credential)', projectId=pid,
                status=latest['status'] if latest else 'idle', task=latest.get('stage',task) if latest else task,
                partIds=sorted(set([p['id'] for p in parts if p.get('projectId')==pid and p.get('ownerRoleId')==role]+
                    [j['partId'] for j in matches if j.get('partId')]))))
    return result
