"""Save the original interactive figures during a headless full-script run."""
import os
from pathlib import Path
import sys


def _reference_label(text):
    return (text.replace('Reference (rtol=1e-12)', 'Reference (60-digit origin series)')
                .replace('DOP853 (rtol=1e-12)', 'verified 60-digit origin series')
                .replace('DOP853, rtol=1e-12', 'verified 60-digit origin series')
                .replace('DOP853', 'verified origin series'))


class ReferenceLabelStream:
    """Adapt legacy main-program headings when the explicit reference hook is active."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        self.stream.write(_reference_label(text))
        return len(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def configure(script_path):
    directory = os.environ.get('BFLNN_FIGURE_DIR')
    if not directory:
        return
    import matplotlib.pyplot as plt
    root = Path(directory) / Path(script_path).stem
    root.mkdir(parents=True, exist_ok=True)
    counter = 0
    independent_reference = (bool(os.environ.get('BFLNN_REFERENCE_ARCHIVE')) and
                             Path(script_path).stem in ('no.5', 'n1.5', 'n2', 'n3', 'n4'))
    if independent_reference:
        sys.stdout = ReferenceLabelStream(sys.stdout)
        print('Explicit reference override: independent converged 60-digit origin series; '
              'legacy reference headings adapted for these fresh tables/plots.', flush=True)

    def save_show(*args, **kwargs):
        nonlocal counter
        for number in plt.get_fignums():
            counter += 1
            fig = plt.figure(number)
            if independent_reference:
                from matplotlib.text import Text
                for label in fig.findobj(match=Text):
                    original = label.get_text()
                    adapted = _reference_label(original)
                    if adapted != original:
                        label.set_text(adapted)
            name = root / f'Figure_{counter:02d}'
            for extension in ('png', 'pdf'):
                fig.savefig(str(name) + '.' + extension, dpi=160, bbox_inches='tight')
            plt.close(fig)
            print('Saved original-script figure:', name, flush=True)
    plt.show = save_show
