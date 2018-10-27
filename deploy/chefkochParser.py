from bs4 import BeautifulSoup, Tag
from urllib.request import urlopen
from pathlib import Path
import json
from copy import copy
import pandas as pd

CHEFKOCH_PATH = 'https://www.chefkoch.de/rezepte/'
PLACEHOLDER = 'https://i.imgur.com/UYTqJ28.gif'

chefkoch_rezepte = pd.read_csv('meta/chefkoch.csv', index_col=False)

with open('meta/links.txt') as f:
    print('Loading JSON')
    json_data = json.load(f)

def get_ingredients(parsed):
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


def get_instructions(parsed):
    instructions_html = parsed.find(id="rezept-zubereitung").strings
    instruction_string = ""
    for instructionHtml in instructions_html:
        instruction_string += instructionHtml
    instructions_split = instruction_string.split(chr(10)) # new line char
    instructions = [x.lstrip() for x in instructions_split if not is_whitespace(x)]
    return instructions

def get_corresponding_recipes(i, food, online):
    '''
        Offline: Serve data from csv files
    '''
    print(food[2], end='-')
    food_info = chefkoch_rezepte.query('recipe_id in @food[2]')
    # TODO: What if dataframe is empty
    if food_info.empty: 
        return list(*zip())
    picture_url_big = json_data[str(food[2])][int(food[3])]
    picture_url_big = picture_url_big.replace('420x280-fix', '960x720')
    picture_url_thumbnail = ''

    recipe_title = food_info['recipe_name'].iloc[0]
    recipe_decs = ''
    instructions = food_info['instructions'].iloc[0]
    mixed = food_info['ingredients'].iloc[0]
    mixed = mixed.split(',')
    ingredients_amount = ''
    ingredients_name = ''
    return i, picture_url_big, picture_url_thumbnail, recipe_title, recipe_decs, ingredients_amount, ingredients_name, instructions, online

def is_whitespace(str):
    return len(str) == 0 or str.isspace()

def generate_html(i, picture_url_big, picture_url_thumbnail, recipe_title, recipe_decs, ingredients_amount, ingredients_name, instructions, online):
    contents = Path("template.html").read_text()
    parsed = BeautifulSoup(contents, 'html.parser')
    instructions_div_container = parsed.select("#cont_text_det_preparation_div")[0]
    instructions_div_element = parsed.select(".cont_text_det_preparation")[0]
    # Title
    parsed.select(".cont_detalles")[0].find('h3').insert(0, recipe_title)
    # Desciption
    parsed.select(".cont_detalles")[0].find('p').insert(0, recipe_decs)
    # Instructions
    schritte = ['SCHRITT '+str(n) for n in list(range(1, len(instructions)+1))]
    
    # Merge every other sentence
    if online is False:
        instructions = instructions.split('.')
        instructions = ['.'.join(x) for x in zip(instructions[0::2], instructions[1::2])]

    for index, instruction in enumerate(instructions):
        instruction_ = copy(instructions_div_element)
        instruction_.findAll('p')[0].insert(0, schritte[index])
        instruction_.findAll('p')[1].insert(0, instruction)
        instructions_div_container.append(instruction_)

    instructions_div_element.decompose()

    # Don't show the expand button if there's no more than two instructions
    if len(instructions) <= 2:
        parsed.select(".cont_btn_mas_dets")[0].decompose()

    if online:
        parsed.select(".cont_img_back")[0].append(BeautifulSoup("<img src=\"" + str(picture_url_thumbnail) + "\"data-src=\"" +str(picture_url_big)+"\">", "html.parser"))
    else:
        parsed.select(".cont_img_back")[0].append(BeautifulSoup("<img src=\"" + str(picture_url_big) + "\"data-src=\"" +str(picture_url_big)+"\">", "html.parser"))
        #parsed.select(".cont_img_back")[0].append(BeautifulSoup("<img src=\"{{url_for(\'custom_static\', filename=\'%s\')}}\">" % str(picture_url_thumbnail), "html.parser"))
    #for j, ingredient_amount in enumerate(ingredients_amount):
     #   table_body.append(BeautifulSoup("<tr><td>" + ingredient_amount
      #                                  + "</td><td>" + ingredients_name[j] + "</td></tr>", "html.parser"))

    #file = open(recipe_title + ".html", "w")
    #file = open("templates/index.html", "w")
    #file.write(parsed.prettify())
    print(recipe_title)
    return parsed.prettify()


def food_list_html(result_list, online=False):
    '''
        online param: wheter to fetch the 
        food data from chefkoch.de or from disk csv file
    '''
    result = []
    if online:
        # ['category', incep_confidence, recipe_id, image_index, image_path]
        for i, food in enumerate(result_list):
            page = urlopen(CHEFKOCH_PATH+food[2])
            parsed = BeautifulSoup(page, 'html.parser')
            ingredients_amount, ingredients_name = get_ingredients(parsed)
            instructions = get_instructions(parsed)
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
            result.append(generate_html(i, picture_url_big, picture_url_thumbnail, recipe_title, recipe_decs, ingredients_amount, ingredients_name, instructions, online))
        return result
    else:
        for i, food in enumerate(result_list):
            result.append(generate_html(*get_corresponding_recipes(i, food, online)))
        return result
    return None
        #picture_url_big, picture_url_thumbnail, recipe_title, recipe_decs, ingredients_amount, ingredients_name, instructions = get_corresponding_recipes(id_food, chefkoch_rezepte)
        #return generate_html(i, picture_url_big, picture_url_thumbnail, recipe_title, recipe_decs, ingredients_amount, ingredients_name, instructions)