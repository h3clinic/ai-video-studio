"""Resume the original Decision Agent from a verified saved director script."""
def run_sizes(upstream, script, client, bus):
    issues=upstream['script_issues'](script,script['prompt'])
    if issues:raise ValueError('Saved script failed current upstream verifier: '+ '; '.join(issues))
    bus.strict=True
    idea=upstream['IdeaModel']('idea',bus,client)
    bus.log('idea','info','Revalidated saved script; no repeat director API call')
    decision=idea.spawn(upstream['DecisionAgent'],'decision',
                        'Original size planning from saved shared brief',brief=script['brief'])
    nouns=[s['noun'] for s in script['subjects'] if s['role'] in ('actor','patient','prop')]
    if not 1<=len(nouns)<=8 or len(nouns)!=len(set(nouns)):
        raise ValueError('One to eight distinct placed subjects required')
    estimates={}
    # Serialize network requests for this bounded integration test. Upstream's
    # actual length(), think(), spawn() and verifier remain unchanged.
    for noun in nouns:
        estimates[noun]=decision.length(noun)
    return dict(estimates=estimates,units='metres',status='size_proposals_verified',
                bindings_verified=False,scene_modified=False,
                limitations=['Model estimates, not measured asset dimensions',
                 'Upstream size verifier checks positive values and integer leg counts, not species anatomy',
                 'Canonical asset alignment and interaction clearance must still be verified'])
