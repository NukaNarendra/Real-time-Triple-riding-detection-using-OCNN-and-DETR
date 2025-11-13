import sys, traceback, inspect
sys.path.insert(0, '.')
from inference.pipeline import InferencePipeline

def call_process(p, video_path):
    # inspect process_video signature
    sig = inspect.signature(p.process_video)
    params = list(sig.parameters.keys())
    print('process_video params:', params)
    # prepare candidate arg bundles
    tries = []
    # simplest: just path
    tries.append(( (video_path,), {} ))
    # common variants:
    tries.append(( (video_path,), {'save_output': True} ))
    tries.append(( (video_path,), {'output_path': 'outputs/', 'save_output': True} ))
    tries.append(( (video_path,), {'save': True} ))
    tries.append(( (video_path,), {'output_dir': 'outputs/'} ))
    tries.append(( (video_path,), {'verbose': True} ))
    tries.append(( (video_path,), {'save_output': True, 'verbose': True} ))
    # try matching params to available names
    for args, kwargs in tries:
        # filter kwargs to only those accepted by the signature
        kwargs_filtered = {k:v for k,v in kwargs.items() if k in params}
        try:
            print(f'Trying call with args={args} kwargs={kwargs_filtered}')
            res = p.process_video(*args, **kwargs_filtered)
            print('\\n? PROCESS SUMMARY (successful call):', res)
            return True
        except TypeError as e:
            print('  -> TypeError:', e)
        except Exception:
            print('  -> Exception during run:')
            traceback.print_exc()
    print('\\n? All automatic call attempts failed. See above.')
    return False

def main():
    try:
        p = InferencePipeline(detector_device='cpu',
                              detector_name='fasterrcnn',
                              min_frames_for_violation=1,
                              rider_threshold=3)
        print('InferencePipeline created, now calling process_video...')
        ok = call_process(p, 'test_video.mp4')
        if not ok:
            print('You can paste the printed process_video params here and I will craft the exact call.')
    except Exception:
        print('Exception creating pipeline:')
        traceback.print_exc()

if __name__ == '__main__':
    main()
