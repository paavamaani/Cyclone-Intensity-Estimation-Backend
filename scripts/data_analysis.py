import pandas as pd

def prepareDataList(res):
  res_dict = res[['year', 'min', 'max']].to_dict(orient = 'list')

  int_min = [0 for i in range(14)]
  int_max = [0 for i in range(14)]

  for i in range(len(res_dict['year'])):
    int_min[res_dict['year'][i] - 9] = res_dict['min'][i]
    int_max[res_dict['year'][i] - 9] = int(res_dict['max'][i])

  return [int_min, int_max]

def getDataAnalysisData(filename):
  month_to_val = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
  }

  df = pd.read_csv('./static/' + 'Data-Analysis.csv')
  df['max'] = pd.to_numeric(df['max'], errors='coerce')

  month = filename.split('_')[1][2:5].lower()
  month = month_to_val[month]

  res = df[df['month'] == month].sort_values(by = 'year')

  res = prepareDataList(res)

  return res