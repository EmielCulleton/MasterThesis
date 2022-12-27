## loading in XML test


content = []
# Read the XML file
with open("sample.xml", "r") as file:
    # Read each line in the file, readlines() returns a list of lines
    content = file.readlines()
    # Combine the lines in the list into a string
    content = "".join(content)
    bs_content = bs(content, "lxml")


#print(content)

url = 'https://rss.nytimes.com/services/xml/rss/nyt/US.xml'
xml_data = requests.get(url).content 

def parse_xml(xml_data):
  # Initializing soup variable
    soup = bs(xml_data, 'xml')

  # Creating column for table
    df = pd.DataFrame(columns=['guid', 'title', 'pubDate', 'description'])

  # Iterating through item tag and extracting elements
    all_items = soup.find_all('item')
    items_length = len(all_items)
    
    for index, item in enumerate(all_items):
        guid = item.find('guid').text
        title = item.find('title').text
        pub_date = item.find('pubDate').text
        description = item.find('description').text

       # Adding extracted elements to rows in table
        row = {
            'guid': guid,
            'title': title,
            'pubDate': pub_date,
            'description': description
        }

        df = df.append(row, ignore_index=True)
        print(f'Appending row %s of %s' % (index+1, items_length))

    return df

df = parse_xml(xml_data)
#print(df.head(5))

with open("cow_sents.txt", "r") as cow:
    content = cow.readlines()
    # Combine the lines in the list into a string
    content = "".join(content)
    bs_content = bs(content, "lxml")