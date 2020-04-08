import logging
from github import Github, ContentFile
from github.Repository import Repository
import pandas as pd
import numpy as np
import glob
import os


from typing import List

GITHUB_TOKEN = os.getenv('GITHUB_PUBLIC_READ_TOKEN')
assert GITHUB_TOKEN is not None

def get_csvs():
    g = Github(login_or_token=GITHUB_TOKEN)
    repo = g.get_repo('CSSEGISandData/COVID-19')
    repo: Repository
    c: List[ContentFile] = repo.get_contents('csse_covid_19_data/csse_covid_19_daily_reports')

    local_csvs = glob.glob('*.csv')

    need_dowload = [f for f in c if f.path.endswith('.csv') and os.path.basename(f.path) not in local_csvs]
    for cf in need_dowload:
        fn = os.path.basename(cf.path)
        logging.info(f'writing {fn}')
        with open(fn, 'wb') as f:
            f.write(cf.decoded_content)

def path_to_date(path):
    path = os.path.basename(path)
    path = os.path.splitext(path)[0]
    date = pd.Timestamp(f'{path[-4:]}-{path[:2]}-{path[3:5]}')
    return date

def df_from_csv(path):
    column_info = {
        'FIPS': '',
        'Admin2': '',
        'Province_State': 'Province/State',
        'Country_Region': 'Country/Region',
        'Last_Update': 'Last Update',
        'Lat': 'Latitude',
        'Long_': 'Longitude',
        'Confirmed': 'Confirmed',
        'Deaths': 'Deaths',
        'Recovered': 'Recovered',
        'Active': '',
        'Combined_Key': '',
    }
    def combined_key(row):
        keys = ['Admin2', 'Province_State', 'Country_Region']
        values = [row[key] for key in keys]
        r = ', '.join((v for v in values if v and pd.notna(v)))
        return r

    df = pd.read_csv(path)
    df['Date'] = path_to_date(path)
    column_names = list(df.columns)
    for column_name, old_column_name in column_info.items():
        if column_name in column_names:
            continue
        if old_column_name:
            if old_column_name in column_names:
                df.rename(columns={old_column_name:column_name}, inplace=True)
                continue
            else:
                df[column_name] = np.nan
                continue
        if column_name in ['FIPS', 'Admin2']:
            df[column_name] = np.nan
        elif column_name == 'Active':
            df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
        elif column_name == 'Combined_Key':
            df['Combined_Key'] = df.apply(combined_key, axis=1)
        else:
            raise NotImplementedError(f'Unexpected column name: {column_name}')

    return df


def main():
    logging.basicConfig(level=logging.DEBUG)
    #get_csvs()

    csv_files = glob.glob('*.csv')
    csv_files.sort(reverse=True)
    df = pd.concat(df_from_csv(csv_file) for csv_file in csv_files)

    germany = df[df['Country_Region'] == 'Germany']
    



if __name__ == '__main__':
    main()
