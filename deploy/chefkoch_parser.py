from bs4 import BeautifulSoup, Tag
from urllib.request import urlopen
from pathlib import Path
import json
from copy import copy
import pandas as pd
from random import choice

class ChefkochParser:
    """Generate HTML from the food result list.

    Attributes:
        result_list: The calculated food list as the result from the neural networks.
    """
    def __init__(self, result_list=[]):
        self.result_list = result_list
        self.chefkoch_path = 'https://www.chefkoch.de/rezepte/'
        self.placeholder = 'https://i.imgur.com/UYTqJ28.gif'
        self.chefkoch_rezepte = pd.read_csv('meta/chefkoch.csv', index_col=False)

        with open('meta/links.txt') as f:
            self.json_links = json.load(f)

    def update_result_list(self, result_list):
        self.result_list = result_list

    def get_ingredients(self, parsed):
        ingredients_table = parsed.find(class_="incredients")
        ingredients_amount = []
        ingredients_name = []
        for ingredient_row in ingredients_table:
            if isinstance(ingredient_row, Tag):
                columns = ingredient_row.find_all('td')
                for j, column in enumerate(columns):
                    parsed_string = ""
                    if column.a:
                        parsed_string = column.a.string.strip().replace("\xa0", " ")
                    else:
                        if column.sup:
                            strings = column.get_text()
                            for s in strings:
                                if s.isalpha():
                                    parsed_string += " "
                                    parsed_string += s
                                else:
                                    parsed_string += s
                        else:
                            parsed_string = column.get_text().strip().replace("\xa0", " ")
                    if j % 2 == 1:
                        ingredients_name.append(parsed_string)
                    else:
                        ingredients_amount.append(parsed_string)
        return ingredients_amount, ingredients_name


    def get_instructions(self, parsed):
        instructions_html = parsed.find(id="rezept-zubereitung").strings
        instruction_string = ""
        for instructionHtml in instructions_html:
            instruction_string += instructionHtml
        instructions_split = instruction_string.split(chr(10)) # new line char
        instructions = [x.lstrip() for x in instructions_split if not self._is_whitespace(x)]
        return instructions

    def _is_whitespace(self, str):
        return len(str) == 0 or str.isspace()

    def get_corresponding_recipes(self, i, food, online):
        '''
            Offline: Serve data from csv files
        '''
        print(food[2], end='-')
        food_info = self.chefkoch_rezepte.query('recipe_id in @food[2]')
        # TODO: What if dataframe is empty
        if food_info.empty:
            return list(*zip())
        picture_url_big = self.json_links[str(food[2])][int(food[3])]
        picture_url_big = picture_url_big.replace('420x280-fix', '960x720')
        picture_url_thumbnail = picture_url_big.replace('960x720', '420x280-fix')
        recipe_title = food_info['recipe_name'].iloc[0]
        recipe_decs = ''
        instructions = food_info['instructions'].iloc[0]
        mixed = food_info['ingredients'].iloc[0]
        mixed = mixed.split(',')
        ingredients_amount = ''
        ingredients_name = ''
        return i, picture_url_big, picture_url_thumbnail, recipe_title, recipe_decs, ingredients_amount, ingredients_name, instructions, online

    def generate_html(self, i, picture_url_big, picture_url_thumbnail, recipe_title, recipe_decs, ingredients_amount, ingredients_name, instructions, online):
        class_names = ['item item--large', 'item item--medium', 'item']
        contents = Path("template.html").read_text()
        parsed = BeautifulSoup(contents, 'html.parser')

        # random size choice
        class_name = choice(class_names)
        # Card Size
        card = parsed.find('div', class_='item')
        card['class'] = class_name
        card['style'] = "background-image: url('{}')".format(str(picture_url_thumbnail))
        # Title
        card.select(".item__details")[0].insert(0, recipe_title)
        
        #card.append(BeautifulSoup("<img src=\"" + str(picture_url_big) + "\"data-src=\"" +str(picture_url_big)+"\">", "html.parser"))
        print(recipe_title)
        return parsed.prettify()


    def food_list_html(self, online=False):
        '''
            online param: wheter to fetch the 
            food data from chefkoch.de or from disk csv file
        '''
        result = []
        if online:
            # ['category', incep_confidence, recipe_id, image_index, image_path]
            for i, food in enumerate(self.result_list):
                page = urlopen(self.chefkoch_path+food[2])
                parsed = BeautifulSoup(page, 'html.parser')
                ingredients_amount, ingredients_name = self.get_ingredients(parsed)
                instructions = self.get_instructions(parsed)
                recipe_title = parsed.find(class_="page-title").string
                try:
                    recipe_decs = parsed.find(class_="summary").string
                except AttributeError:
                    recipe_decs=''
                # TODO: Prep time
                #meta = parsed.find(id="preparation-info").contents[2].strip()
                picture_url = parsed.find(class_="slideshow-image").attrs['src']
                try:
                    picture_url = picture_url.replace('alt=\"\"/', '')
                    picture_url_big = picture_url.replace('420x280-fix', '960x720')
                    picture_url_thumbnail = picture_url.replace('420x280-fix', 'small')
                except:
                    pass
                result.append(self.generate_html(i, picture_url_big, picture_url_thumbnail, recipe_title, recipe_decs, ingredients_amount, ingredients_name, instructions, online))
            return result
        else:
            for i, food in enumerate(self.result_list):
                try:
                    result.append(self.generate_html(*self.get_corresponding_recipes(i, food, online)))
                except:
                    print('RECIPE NOT FOUND', food[2])
            return result
        return None