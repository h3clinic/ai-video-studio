"""Read-only collection access diagnostic with credential redaction."""
import json
from cloud.runpod_client import pod_request, RunPodError

def main():
    try:
        data=pod_request()
        print(json.dumps({'success':True,'pods':[{k:p.get(k) for k in ('id','name','desiredStatus')} for p in data]}))
    except RunPodError as e:
        print(json.dumps({'success':False,**e.public()}))

if __name__=='__main__':main()
