"""Recorded DAVIS animal clips for object-conditioned Gaussian control graphs."""
import argparse
import json
from pathlib import Path,PurePosixPath
import shutil
import time
import zipfile
import requests
from .checkpoint_io import digest,keep_windows_awake

WORK=Path('../../work/real_video/davis_animals')
URL='https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
TRAIN=['bear','cat-girl','dog-agility','dog-gooses','dogs-scale','elephant','koala','rhino','sheep']
VAL=['camel','cows','goat','pigs']


def download():
    WORK.mkdir(parents=True,exist_ok=True); archive=WORK/'DAVIS-2017-trainval-480p.zip'
    if not archive.exists():
        part=archive.with_suffix('.zip.partial'); offset=part.stat().st_size if part.exists() else 0
        headers={'Range':f'bytes={offset}-'} if offset else {}
        with requests.get(URL,headers=headers,stream=True,timeout=(20,60)) as response:
            response.raise_for_status()
            if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                raise ValueError('Server did not honor resumable download; preserve partial')
            if offset+int(response.headers.get('Content-Length','0'))>900_000_000: raise ValueError('Unexpected download size')
            last=time.monotonic()
            with part.open('ab' if offset else 'xb') as stream:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk); offset+=len(chunk)
                    if offset>900_000_000: raise ValueError('Unexpected download size')
                    if time.monotonic()-last>10:
                        print(json.dumps(dict(stage='download',bytes=offset)),flush=True); last=time.monotonic()
        if part.stat().st_size!=832766765: raise ValueError('Unexpected official archive size')
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None: raise ValueError('Corrupt archive')
        part.rename(archive)
    root=(WORK/'extracted').resolve(); root.mkdir(exist_ok=True)
    selected=set(TRAIN+VAL); count=0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts=PurePosixPath(info.filename).parts
            image=len(parts)==5 and parts[0]=='DAVIS' and parts[1] in ['JPEGImages','Annotations'] and parts[2]=='480p' and parts[3] in selected and PurePosixPath(parts[-1]).suffix.lower() in ['.jpg','.png']
            metadata=len(parts)==4 and parts[:3]==('DAVIS','ImageSets','2017') and parts[-1] in ['train.txt','val.txt']
            if not (image or metadata) or info.is_dir(): continue
            target=root.joinpath(*parts).resolve()
            if not target.is_relative_to(root) or info.file_size>10_000_000: raise ValueError('Unsafe archive member')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with z.open(info) as source,target.open('xb') as dest: shutil.copyfileobj(source,dest)
            if target.stat().st_size!=info.file_size: raise ValueError('Incomplete extracted member')
            count+=1
    official=root/'DAVIS/ImageSets/2017'
    assert set(TRAIN)<=set((official/'train.txt').read_text().split())
    assert set(VAL)<=set((official/'val.txt').read_text().split())
    record=dict(url=URL,archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,
                selected_train_sequences=TRAIN,selected_validation_sequences=VAL,extracted_files=count,
                split_note='Official train/val subsets, disjoint named sequences. All dog sequences kept out of validation; horsejump pairs excluded. No claim of verified identity-level independence.',
                reference='https://davischallenge.org/davis2017/code.html',
                code_terms_reference='https://github.com/fperazzi/davis-2017/blob/main/LICENSE',
                limits='Small animal/mixed-object subset. No downloaded code executed. Other official validation/test sequences not extracted.')
    (WORK/'provenance.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=['download']); parser.parse_args()
    with keep_windows_awake(): download()


if __name__=='__main__': main()
