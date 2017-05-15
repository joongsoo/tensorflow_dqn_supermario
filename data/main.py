__author__ = 'justinarmstrong'

from . import setup,tools
from .states import main_menu,load_screen,level1
from . import constants as c
import pygame as pg
from pygame.surfarray import pixels3d

def main():
    """Add states to control here."""
    run_it = tools.Control(setup.ORIGINAL_CAPTION)
    state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.TIME_OUT: load_screen.TimeOut(),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.LEVEL1: level1.Level1()}

    run_it.setup_states(state_dict, c.LEVEL1)
    #state_dict[c.MAIN_MENU].startup()
    #run_it.main()
    while not run_it.done:
        run_it.event_loop(None)
        run_it.update()
        pg.display.update()
        if run_it.ml_done:
            run_it = tools.Control(setup.ORIGINAL_CAPTION)
            state_dict = {c.MAIN_MENU: main_menu.Menu(),
                          c.LOAD_SCREEN: load_screen.LoadScreen(),
                          c.TIME_OUT: load_screen.TimeOut(),
                          c.GAME_OVER: load_screen.GameOver(),
                          c.LEVEL1: level1.Level1()}

            run_it.setup_states(state_dict, c.LEVEL1)


        run_it.clock.tick(run_it.fps)
        if run_it.show_fps:
            fps = run_it.clock.get_fps()
            with_fps = "{} - {:.2f} FPS".format(run_it.caption, fps)
            pg.display.set_caption(with_fps)



