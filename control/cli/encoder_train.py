from pathlib import Path
import sys
import os
import webbrowser
# Ensure repo root is importable and DLL priority is set early
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    import ensure_dlls
    ensure_dlls.ensure_dll_priority()
except Exception:
    pass

from utils.argutils import print_args
from models.encoder.train import train
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the speaker encoder. You must have run encoder_preprocess.py first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance. If a model state from the same run ID was previously "
        "saved, the training will restart from there. Pass -f to overwrite saved states and "
        "restart from scratch.")
    parser.add_argument("clean_data_root", type=Path, help= \
        "Path to the output directory of encoder_preprocess.py. If you left the default "
        "output directory when preprocessing, it should be <datasets_root>/SV2TTS/encoder/.")
    parser.add_argument("-m", "--models_dir", type=Path, default="encoder/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    parser.add_argument("-v", "--vis_every", type=int, default=10, help= \
        "Number of steps between updates of the loss and the plots.")
    parser.add_argument("-u", "--umap_every", type=int, default=100, help= \
        "Number of steps between updates of the umap projection. Set to 0 to never update the "
        "projections.")
    parser.add_argument("-s", "--save_every", type=int, default=500, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=7500, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model.")
    parser.add_argument("--visdom_server", type=str, default="http://localhost")
    parser.add_argument("--no_visdom", action="store_true", help= \
        "Disable visdom.")
    parser.add_argument("--start_visdom", action="store_true", help="Start a local visdom server before training")
    parser.add_argument("--visdom_port", type=int, default=8097, help="Port for visdom server")
    parser.add_argument("--visdom_host", type=str, default="127.0.0.1", help="Hostname for visdom server")
    args = parser.parse_args()
    
    # Process the arguments
    args.models_dir.mkdir(exist_ok=True)
    
    # Run the training
    print_args(args, parser)
    auto_visdom = os.environ.get("MOCKINGBIRD_AUTO_VISDOM", "1").strip().lower() not in {"0", "false", "no"}
    should_start_visdom = auto_visdom or getattr(args, "start_visdom", False)
    if should_start_visdom and not getattr(args, "no_visdom", False):
        try:
            from tools.visdom_helper import start_visdom_server
            started = start_visdom_server(port=args.visdom_port, host=args.visdom_host)
            if started:
                print(f"Started visdom server at {args.visdom_host}:{args.visdom_port}")
                try:
                    webbrowser.open(f"http://{args.visdom_host}:{args.visdom_port}")
                except Exception:
                    pass
            else:
                print(f"Failed to start visdom server at {args.visdom_host}:{args.visdom_port}")
        except Exception as e:
            print(f"Warning: could not start visdom helper: {e}")

    train(**vars(args))
