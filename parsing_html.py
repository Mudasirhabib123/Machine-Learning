from bs4 import BeautifulSoup

html = """
<div class='full_name'><span style='font-weight:bold'>
Masego</span> Azra</div>"
"""
soup = BeautifulSoup(html ,'lxml')
soup = soup.find('div',{'class' : 'full_name'}).text

print(soup)