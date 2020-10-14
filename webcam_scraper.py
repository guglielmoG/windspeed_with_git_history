from selenium import webdriver
from datetime import datetime
import time
import os

class WebcamScraper:

    # from selenium import webdriver
    # from datetime import datetime
    # import time
    # import os

    def __init__(self, b_dir = '', txt_file = 'webcam_links.txt'):

        self.b_dir = b_dir
        self.path_to_txt = txt_file
        self.webcams = {}

    def load_webcams(self):
        with open(self.path_to_txt) as f:
            for line in f:
                (key, val) = line.split()
                self.webcams[key] = val

    def get_data(self, folder_name = 'data_raw'):
        self.load_webcams()
        driver = webdriver.Firefox(executable_path = os.path.join(self.b_dir, 'geckodriver') )
        for city, link in self.webcams.items():
            driver.get(link)
            now = datetime.now()
            rel_path = folder_name + '/' + city + '_' + now.strftime("%d_%m_%Y__%H_%M_%S") +'.png'
            driver.get_screenshot_as_file(os.path.join(self.b_dir, rel_path))
            
        driver.close()


    def automatic_scraper(self, interval, iteration):
        for i in range(iteration):
            self.get_data()
            time.sleep(interval*60)
