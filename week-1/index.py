import http.client
import json
import csv

base_path = '/search/v4.3/all/results?cateid=DSAA31'

def fetch_asus_page(page):
    connection = http.client.HTTPSConnection('ecshweb.pchome.com.tw')
    connection.request('GET', f'{base_path}&page={page}')
    response = connection.getresponse()
    return json.loads(response.read().decode('utf-8'))

first_data = fetch_asus_page(1)

prods = first_data['Prods']
total_page = first_data['TotalPage']

if total_page != 1:
    for i in range(2, total_page + 1):
        prods.extend(fetch_asus_page(i)['Prods'])

info_of_i5_pc = {
    'total_price' : 0,
    'total_count' : 0
}

info_of_all_pc = {
    'total_price' : 0,
    'total_count' : len(prods)
}

for prod in prods:
    prod_price = prod['Price']

    info_of_all_pc['total_price'] += prod_price

    if 'i5' in prod['Name']:
        info_of_i5_pc['total_price'] += prod_price
        info_of_i5_pc['total_count'] += 1



avg_price_i5 = info_of_i5_pc['total_price'] / info_of_i5_pc['total_count']
avg_price_all = info_of_all_pc['total_price'] / info_of_all_pc['total_count']

print(f'i5 price avg: {avg_price_i5}')

squared_diff_sum = 0
for prod in prods:
    squared_diff_sum += (prod['Price'] - avg_price_all)**2

std = (squared_diff_sum / info_of_all_pc['total_count']) ** 0.5
    
with open('products.txt', 'w') as products_file, \
     open('best_products.txt', 'w') as best_products_file, \
     open('standardization.csv', mode='w', newline='') as csv_file:    
    writer = csv.writer(csv_file)
    for prod in prods:
        prod_id = prod['Id']
        prod_price = prod['Price']
        prod_rating_value = prod['ratingValue']
        prod_review_count = prod['reviewCount']

        products_file.write(prod_id + '\n')

        if prod_rating_value and prod_review_count and prod_rating_value > 4.9 and prod_review_count > 0:
            best_products_file.write(prod_id + '\n')
        
        price_z_score = (prod_price - avg_price_all) / std
        writer.writerow([prod_id, prod_price, price_z_score])
