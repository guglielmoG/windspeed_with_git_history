from selenium import webdriver
from datetime import datetime
import time
import os


webcams = {'bassano':'http://www.meteocimagrappa.it/sudest/webcam2.php?v=55832%27',
           'albisola': 'https://lnialbisola.wdsitems.it/cam_display.php?station=lnialbisola&element=cam',
           'monfalcone': 'https://amministrazioneaperta.comune.monfalcone.go.it/windEth-jpg.php?rnd=0.5469449876751713',
           'riccione': 'http://www.windsurf-maniac.it/web-cam-151/web-cam.jpg',
           'rimini': 'https://www.meteogiuliacci.it/webcam/rimini.jpg',
           'torbole': 'http://www.addicted-sports.com/fileadmin/webcam/torbole/current/1200.jpg',
           'malecesine': 'http://www.addicted-sports.com/fileadmin/webcam/gardasee/current/1200.jpg',
           'thuile': 'http://www.meteolathuile.com/piccolo_san_bernardo_sud.jpg',
           'thuile2': 'http://www.meteolathuile.com/piccolo_san_bernardo.jpg',
           'campegli': 'https://www.meteo-lazio.it/webcams/campaegli/FI9901EP_00626E8B72EB/snap/webcam.php'
           }




class WebcamScraper:

    def __init__(self,webcams, b_dir = ''):

        self.webcam_location = [location for location in webcam_link]
        self.webcam_links[link for link in webcams.values()]
        self.b_dir = b_dir

    def mk_path(self, rel_path) :
        return os.path.join(self.b_dir, rel_path)


    def time_string(self):
        now = datetime.now()
        return now.strftime("%d_%m_%Y__%H_%M_%S")

    def run(self, lag = 10, n_image = 10, name_folder = 'data_row'):
        """
        lag: the interval between each screenshot expressed in minutes
        n_image: is the total number of screenshot that we want to scraper
        name_folder: is the name of the subdirectory in which we want to save
        the screenshots
        """
        t = 0
        while True:
            for i in range(len(self.webcam_links)):

                driver = webdriver.Firefox(executable_path = self.mk_path('geckodriver'))
                driver.get(self.webcam_links[i])
                location = self.webcam_location[i]
                data_time = self.time_string()
                name = '_' + location + '_' + data_time + '.png'
                driver.get_screenshot_as_file(self.mk_path(name_folder+'/'+name))
                t += 1

                if t == n_image:
                    self.driver.close()
                    return

            time.sleep(lag*60)
