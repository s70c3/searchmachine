import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib_backend import MatplotlibBackend
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import io

DPI = 100
BG_COLOR = 255
MARGIN = 2

def _fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf) #dpi=DPI
    buf.seek(0)
    img = Image.open(buf)
    return img


def dxf2padded_png(path):#, newpath):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    auditor = doc.audit()
    if len(auditor.errors):
        print('[WARN] %d errors in dxf file %s' % (len(auditor.errors, path)))
    
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    ctx.set_current_layout(msp)
    ctx.current_layout.set_colors(bg='#FFFFFF')
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)

    img = _fig2img(fig)
    plt.close()
    return img


def cut_png(img):
#     img = Image.open(path).convert('L')
    img = np.array(img)

    x = (img != BG_COLOR).sum(axis=1)
    y = (img != BG_COLOR).sum(axis=0)

    pad_top = np.where(x != 0)[0][0] - MARGIN
    pad_bot = np.where(x != 0)[0][-1] +MARGIN
    pad_left= np.where(y != 0)[0][0]  -MARGIN
    pad_right=np.where(y != 0)[0][-1] +MARGIN
    
    img = img[pad_top:pad_bot, pad_left:pad_right]
    img = Image.fromarray(img)
    return img

def dxf2image(path, scale_koef=1.0):
    img = dxf2padded_png(path)#, newpath)
    img = cut_png(img)
    
    newsize = tuple(map(lambda s: int(s*scale_koef), img.size))
    img = img.resize(newsize)
    
    # after resizing images with lines width=1 has very low contrast
    # enlarge contrast manually
    img = ImageEnhance.Contrast(img)
    img = img.enhance(100.0)
    img.save('pic.png')
    
    return img

if __name__ == "__main__":
    old_path = './24.00.14.002.dxf'
    path = old_path.replace('.dxf', '.png')

    dxf2image(old_path)

