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
                (load_check, key, val) = line.split()
                if load_check == 'y':
                    self.webcams[key] = val


    def automatic_scraper(self, interval, iteration):
        t0=time.time()
        def get_data(folder_name = 'data_raw'):
            for city, link in self.webcams.items():
                driver.get(link)
                now = datetime.now()
                rel_path = folder_name + '/' + city + '_' + now.strftime("%d_S%m_%Y__%H_%M_%S") +'.png'
                driver.get_screenshot_as_file(os.path.join(self.b_dir, rel_path))

        self.load_webcams()
        # chrome_options = webdriver.ChromeOptions()
        # chrome_options.add_argument('--headless')
        # driver = webdriver.Chrome(options=chrome_options,executable_path = os.path.join(self.b_dir, 'chromedriver') )
        firefox_options=webdriver.FirefoxOptions()
        firefox_options.add_argument('--headless')
        driver = webdriver.Firefox(options=firefox_options,executable_path = os.path.join(self.b_dir, 'geckodriver') )
        for i in range(iteration):
            get_data()
            time.sleep(interval*60)
        driver.close()
        print(time.time()-t0)
