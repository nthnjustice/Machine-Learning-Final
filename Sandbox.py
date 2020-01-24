import urllib
import zipfile
import geopandas as gpd
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

countyList = open('Data/CountyList.txt', 'r+')
countyList = countyList.read().split('\n')

countyList = [i.replace(" ", "").lower().capitalize() for i in countyList]

countyList = countyList[1:4]

base = 'ftp://ftp.geoinfo.msl.mt.gov/Data/Spatial/MSDI/Cadastral/Parcels/'

for i in countyList:
    url = base + i + '/' + i + 'OwnerParcel_shp.zip'
    res = urllib.request.urlopen(url).read()

    with open('Data/Parcels/' + i + '.zip', 'wb') as f:
        f.write(res)
    f.close()

for i in countyList:
    zip_ref = zipfile.ZipFile('Data/Parcels/' + i + '.zip', 'r')
    zip_ref.extractall('Data/Parcels/' + i)
    zip_ref.close()

parcels = {}
base = 'Data/Parcels/'

for i in countyList:
    path = base + i + '/' + i + 'OwnerParcel_shp.shp'
    parcels[i] = gpd.read_file(path)

res = urllib.request.urlopen('ftp://ftp.geoinfo.msl.mt.gov/Data/Spatial/MSDI/Cadastral/PublicLands/MTPublicLands_SHP.zip').read()

with open('Data/PublicLands/PublicLands.zip', 'wb') as f:
    f.write(res)
f.close()

zip_ref = zipfile.ZipFile('Data/PublicLands/PublicLands.zip', 'r')
zip_ref.extractall('Data/PublicLands/PublicLands')
zip_ref.close()

publicLands = gpd.read_file('Data/PublicLands/PublicLands/MTPublicLands_SHP.shp')

res = urllib.request.urlopen('ftp://ftp.geoinfo.msl.mt.gov/Data/Spatial/MSDI/Cadastral/ConservationEasements/MTConEasements_SHP.zip').read()

with open('Data/ConservationEasement/ConservationEasement.zip', 'wb') as f:
    f.write(res)
f.close()

zip_ref = zipfile.ZipFile('Data/ConservationEasement/ConservationEasement.zip', 'r')
zip_ref.extractall('Data/ConservationEasement/ConservationEasement')
zip_ref.close()

conservationEasement = gpd.read_file('Data/ConservationEasement/ConservationEasement/MTConEasements_SHP.shp')

adminBoundaryList = open('Data/AdminBoundaryList.txt', 'r+')
adminBoundaryList = adminBoundaryList.read().split('\n')

base = 'ftp://ftp.geoinfo.msl.mt.gov/Data/Spatial/MSDI/AdministrativeBoundaries/'

for i in adminBoundaryList:
    url = base + 'Montana' + i + '_shp.zip'
    res = urllib.request.urlopen(url).read()

    with open('Data/AdministrativeBoundaries/' + i + '.zip', 'wb') as f:
        f.write(res)
    f.close()

for i in adminBoundaryList:
    zip_ref = zipfile.ZipFile('Data/AdministrativeBoundaries/' + i + '.zip', 'r')
    zip_ref.extractall('Data/AdministrativeBoundaries/' + i)
    zip_ref.close()

adminBoundaries = {}
base = 'Data/AdministrativeBoundaries/'

adminBoundaries['Counties'] = gpd.read_file(base + 'Counties/MontanaCounties_shp/County.shp')
adminBoundaries['IncorporatedCitiesTowns'] = gpd.read_file(base + 'IncorporatedCitiesTowns/MontanaIncorporatedCitiesTowns_shp/MontanaIncorporatedCitiesTowns.shp')
adminBoundaries['ManagedAreas'] = gpd.read_file(base + 'ManagedAreas/MontanaManagedAreas.shp')
adminBoundaries['Reservations'] = gpd.read_file(base + 'Reservations/MontanaReservations_shp/MontanaReservations.shp')
adminBoundaries['SchoolDistricts'] = gpd.read_file(base + 'SchoolDistricts/MontanaSchoolDistricts_shp/Unified.shp')
adminBoundaries['StateBoundary'] = gpd.read_file(base + 'StateBoundary/MontanaStateBoundary_shp/StateofMontana.shp')
adminBoundaries['TIFDs'] = gpd.read_file(base + 'TIFDs/MontanaTIFDs_shp/TIFD.shp')
adminBoundaries['WeedDistricts'] = gpd.read_file(base + 'WeedDistricts/MontanaWeedDistricts_shp/MontanaWeedDistricts.shp')

res = urllib.request.urlopen('http://ftp.geoinfo.msl.mt.gov/Data/Spatial/MSDI/AdministrativeBoundaries/NationalParkServiceAdminBoundaries_shp.zip').read()

with open('Data/NationalParkServiceAdmin/NationalParkServiceAdminBoundaries_shp.zip', 'wb') as f:
    f.write(res)
f.close()

zip_ref = zipfile.ZipFile('Data/NationalParkServiceAdmin/NationalParkServiceAdminBoundaries_shp.zip', 'r')
zip_ref.extractall('Data/NationalParkServiceAdmin/')
zip_ref.close()

npsAdmin = gpd.read_file('Data/NationalParkServiceAdmin/NationalParkServiceAdminBoundaries_shp/NationalParkServiceAdminBoundaries_Montana.shp')

def fetchPolygonCoords(poly):
    if poly.type == 'Polygon':
        exterior = poly.exterior.coords[:]
        interior = []

        for i in poly.interiors:
            interior += i.coords[:]
    elif poly.type == 'MultiPolygon':
        exterior = []
        interior = []

        for i in poly:
            fpc = fetchPolygonCoords(i)
            exterior += fpc['exterior']
            interior += fpc['interior']

    return {'exterior': exterior, 'interior': interior}

def countCoordinates(poly):
    coords = fetchPolygonCoords(poly)

    return (len(coords['exterior']), len(coords['interior']))

def isMultiPolygon(poly):
    if poly.type == 'MultiPolygon':
        return 1
    else:
        return 0

managedAreas = adminBoundaries['ManagedAreas']
blm = managedAreas[managedAreas['INST'] == 'US Bureau of Land Management']
blm.reset_index(drop=True, inplace=True)
managedAreas =  managedAreas[managedAreas['INST'] != 'US Bureau of Land Management']
managedAreas.reset_index(drop=True, inplace=True)
adminBoundaries['ManagedAreas'] = managedAreas

inputs = {'BLM': [], 'Area': [], 'Perimeter': [],
          'Exterior Coords': [], 'MultiPolygon': [], 'Interior Coords': [],
          'Convex Hull Dist': []}

for i in range(blm.shape[0]):
    coords = countCoordinates(blm.loc[i, 'geometry'])
    convexHull = blm.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(1)
    inputs['Area'].append(blm.loc[i, 'Shape_Area'])
    inputs['Perimeter'].append(blm.loc[i, 'Shape_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(blm.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

blm = publicLands[publicLands['OWNER'] == 11]
blm.reset_index(drop=True, inplace=True)
publicLands = publicLands[publicLands['OWNER'] != 11]
publicLands.reset_index(drop=True, inplace=True)

for i in range(blm.shape[0]):
    coords = countCoordinates(blm.loc[i, 'geometry'])
    convexHull = blm.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(1)
    inputs['Area'].append(blm.loc[i, 'Shape_STAr'])
    inputs['Perimeter'].append(blm.loc[i, 'Shape_STLe'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(blm.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

for i in range(managedAreas.shape[0]):
    coords = countCoordinates(managedAreas.loc[i, 'geometry'])
    convexHull = managedAreas.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(managedAreas.loc[i, 'Shape_Area'])
    inputs['Perimeter'].append(managedAreas.loc[i, 'Shape_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(managedAreas.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

for i in range(publicLands.shape[0]):
    coords = countCoordinates(publicLands.loc[i, 'geometry'])
    convexHull = publicLands.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(publicLands.loc[i, 'Shape_STAr'])
    inputs['Perimeter'].append(publicLands.loc[i, 'Shape_STLe'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(publicLands.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

counties = adminBoundaries['Counties']

for i in range(counties.shape[0]):
    coords = countCoordinates(counties.loc[i, 'geometry'])
    convexHull = counties.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(counties.loc[i, 'Shape_Area'])
    inputs['Perimeter'].append(counties.loc[i, 'Shape_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(counties.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

incorporated = adminBoundaries['IncorporatedCitiesTowns']

for i in range(incorporated.shape[0]):
    coords = countCoordinates(incorporated.loc[i, 'geometry'])
    convexHull = incorporated.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(incorporated.loc[i, 'Shape_Area'])
    inputs['Perimeter'].append(incorporated.loc[i, 'Shape_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(incorporated.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

reservations = adminBoundaries['Reservations']

for i in range(reservations.shape[0]):
    coords = countCoordinates(reservations.loc[i, 'geometry'])
    convexHull = reservations.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(reservations.loc[i, 'SHAPE_Area'])
    inputs['Perimeter'].append(reservations.loc[i, 'SHAPE_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(reservations.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

schools = adminBoundaries['SchoolDistricts']

for i in range(schools.shape[0]):
    coords = countCoordinates(schools.loc[i, 'geometry'])
    convexHull = schools.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(schools.loc[i, 'SHAPE_Area'])
    inputs['Perimeter'].append(schools.loc[i, 'SHAPE_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(schools.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

state = adminBoundaries['StateBoundary']

for i in range(state.shape[0]):
    coords = countCoordinates(state.loc[i, 'geometry'])
    convexHull = state.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(state.loc[i, 'SHAPE_Area'])
    inputs['Perimeter'].append(state.loc[i, 'SHAPE_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(state.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

tifds = adminBoundaries['TIFDs']

for i in range(tifds.shape[0]):
    coords = countCoordinates(tifds.loc[i, 'geometry'])
    convexHull = tifds.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(tifds.loc[i, 'SHAPE_Area'])
    inputs['Perimeter'].append(tifds.loc[i, 'SHAPE_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(tifds.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

for i in range(conservationEasement.shape[0]):
    coords = countCoordinates(conservationEasement.loc[i, 'geometry'])
    convexHull = conservationEasement.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(conservationEasement.loc[i, 'Shape_STAr'])
    inputs['Perimeter'].append(conservationEasement.loc[i, 'Shape_STLe'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(conservationEasement.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

for i in range(npsAdmin.shape[0]):
    coords = countCoordinates(npsAdmin.loc[i, 'geometry'])
    convexHull = npsAdmin.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(npsAdmin.loc[i, 'Shape_Area'])
    inputs['Perimeter'].append(npsAdmin.loc[i, 'Shape_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(npsAdmin.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

for i in countyList:
  county = parcels[i]

  for i in range(county.shape[0]):
    coords = countCoordinates(county.loc[i, 'geometry'])
    convexHull = county.loc[i, 'geometry'].convex_hull
    maxHull = max(convexHull.bounds)
    minHull = min(convexHull.bounds)

    inputs['BLM'].append(0)
    inputs['Area'].append(county.loc[i, 'SHAPE_Area'])
    inputs['Perimeter'].append(county.loc[i, 'SHAPE_Leng'])
    inputs['Exterior Coords'].append(coords[0])
    inputs['MultiPolygon'].append(isMultiPolygon(county.loc[i, 'geometry']))
    inputs['Interior Coords'].append(coords[1])
    inputs['Convex Hull Dist'].append(maxHull - minHull)

inputs = pd.DataFrame(inputs)

X = inputs.drop(columns=['BLM'])
y = inputs['BLM']

featureNames = X.columns

ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

pipe_rf = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestClassifier', RandomForestClassifier())])

pipe_rf.fit(X, y)

f_importances = pd.Series(pipe_rf.named_steps['RandomForestClassifier'].feature_importances_, featureNames)
f_importances = f_importances.sort_values(ascending=False)

#f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16,9), rot=45, fontsize=30)
#plt.tight_layout()
#plt.show()

pipe_lr = Pipeline([('StandardScaler', StandardScaler()), ('LogisticRegression', LogisticRegression())])

pipe_lr.fit(X, y)

y_pred = pipe_lr .predict(X_test)
y_pred_score = pipe_lr .predict_proba(X_test)

print("B2 accuracy:", str(accuracy_score(y_test, y_pred) * 100))
print("B2 ROC AUC:", str(roc_auc_score(y_test, y_pred_score[:,1]) * 100))

accs = cross_val_score(pipe_lr,
                       X,
                       y,
                       cv=KFold(n_splits=10, random_state=0))

print('The average score of the model: ', round(accs.mean(), 3))
print('The std score of the accuracy: ', round(accs.std(), 3))

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['StatusClass'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
  yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.savefig('Data/Model/Output/1997-2012-(WithVHF)/CM(full).png', bbox_inches='tight', pad_inches=0.25)

inputs.head()
