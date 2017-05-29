__author__ = 'justinarmstrong'

"""
This module initializes the display and creates dictionaries of resources.
"""


import platform

p_name = platform.system()
print p_name

import os
import pygame as pg
from . import tools
from . import constants as c
ORIGINAL_CAPTION = c.ORIGINAL_CAPTION


os.environ['SDL_VIDEO_CENTERED'] = '1'
pg.init()
pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
pg.display.set_caption(c.ORIGINAL_CAPTION)
SCREEN = pg.display.set_mode(c.SCREEN_SIZE, 0, 32)
SCREEN_RECT = SCREEN.get_rect()
FONTS = tools.load_all_fonts(os.path.join("resources", "fonts"))
MUSIC = tools.load_all_music(os.path.join("resources", "music"))
GFX = tools.load_all_gfx(os.path.join("resources", "graphics"))
SFX = tools.load_all_sfx(os.path.join("resources", "sound"))
# dev env
'''
if p_name == "Darwin":
    import os
    import pygame as pg
    from . import tools
    from . import constants as c
    ORIGINAL_CAPTION = c.ORIGINAL_CAPTION


    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
    pg.display.set_caption(c.ORIGINAL_CAPTION)
    SCREEN = pg.display.set_mode(c.SCREEN_SIZE, 0, 32)
    SCREEN_RECT = SCREEN.get_rect()
    FONTS = tools.load_all_fonts(os.path.join("resources", "fonts"))
    MUSIC = tools.load_all_music(os.path.join("resources", "music"))
    GFX = tools.load_all_gfx(os.path.join("resources", "graphics"))
    SFX = tools.load_all_sfx(os.path.join("resources", "sound"))
# aws
else:
    import os
    # import pygame as pg
    from . import tools
    from . import constants as c

    ORIGINAL_CAPTION = c.ORIGINAL_CAPTION

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame as pg

    pg.init()

    pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
    pg.display.set_caption(c.ORIGINAL_CAPTION)
    SCREEN = pg.display.set_mode(c.SCREEN_SIZE, 0, 32)
    SCREEN_RECT = SCREEN.get_rect()
    FONTS = tools.load_all_fonts(os.path.join("resources", "fonts"))
    MUSIC = tools.load_all_music(os.path.join("resources", "music"))
    GFX = tools.load_all_gfx(os.path.join("resources", "graphics"))
    SFX = tools.load_all_sfx(os.path.join("resources", "sound"))

'''



