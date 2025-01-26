import numpy as np
import pandas as pd
import cv2
import redis
import os
from insightface.app import FaceAnalysis
from datetime import datetime
import time
from sklearn.metrics import pairwise

r = redis.Redis(
    host='redis-17999.c341.af-south-1-1.ec2.redns.redis-cloud.com',
    port = '17999',
    password='ljJG3Iqp6voDDvTsy1SBG1Dm9qVEofk5'
)

# Retrieve Data from Database
def retrieve_data():
    retrieve_dict = r.hgetall(name='NHS-free-db')
    if not retrieve_dict:
        print("No data found in Redis.")
        return pd.DataFrame(columns=['Name', 'Role', 'Facial Features', 'Course'])  # Include 'Course' column

    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda z: np.frombuffer(z, dtype=np.float32))

    index = retrieve_series.index
    index = list(map(lambda q: q.decode(), index))
    retrieve_series.index = index

    retrieve_df = retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['Name_Role', 'Facial Features']

    # Add a debug statement to print the initial DataFrame
    print("Initial DataFrame after retrieval and conversion:")
    print(retrieve_df.head())

    # Filter out rows where the split operation doesn't return exactly 2 elements
    retrieve_df = retrieve_df[retrieve_df['Name_Role'].apply(lambda x: len(x.split('@')) == 2)]

    # Apply the split
    retrieve_df[['Name', 'Role']] = retrieve_df['Name_Role'].apply(lambda x: x.split('@')).tolist()

    # Add a placeholder 'Course' column
    retrieve_df['Course'] = 'Unknown'

    return retrieve_df[['Name', 'Role', 'Facial Features', 'Course']]

# Call the function to see the output
print(retrieve_data())


# Call the function to see the output
print(retrieve_data())





#Configur face analysis
faceapp = FaceAnalysis(name='buffalo_l', root='buffalo_l',providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640), det_thresh=0.5)

# ML search algorithgm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def searchnet(dataf, feature_col, test_vector, name_role_cos=['Name', 'Role'], thresh=0.5):
    dff = dataf.copy()
    xx_list = dff[feature_col].tolist()
    XX = np.asarray(xx_list)

    cos_sim = cosine_similarity(XX, test_vector.reshape(1, -1))
    cos_sim_flat = np.asarray(cos_sim).flatten()
    dff['Cosine'] = cos_sim_flat

    stingray = dff.query(f'Cosine >= {thresh}')
    stingray.reset_index(drop=True, inplace=True)
    if len(stingray) > 0:
        argMX = stingray['Cosine'].idxmax()
        name_found, role_found = stingray.loc[argMX][name_role_cos]
    else:
        name_found = 'Unknown'
        role_found = 'Unknown'
    return name_found, role_found

# To save the logs for every 1 min
class RealTimer:
    def __init__(self):
        self.logs = dict(Name=[], Role=[],Course=[], current_time=[])

    def reset(self):
        self.logs = dict(Name=[], Role=[],Course=[], current_time=[])

    def save_log_redis(self):
        dff = pd.DataFrame(self.logs)
        dff.drop_duplicates('Name', inplace=True)
        namelist = dff['Name'].tolist()
        rolelist = dff['Role'].tolist()
        courselist = dff['Course'].tolist()
        ctimelist = dff['current_time'].tolist()
        encoded_data = []

        for name, role, course, ctime in zip(namelist, rolelist, courselist, ctimelist):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{course}@{ctime}"
                encoded_data.append(concat_string)
        if len(encoded_data) > 0:
            r.lpush('NHS-free-db:logs', *encoded_data)
        self.reset()


    def face_prediction(self, photo, dataf, feature_col, name_role_cos=['Name', 'Role'], course = 'Unknown',thresh=0.5):
        current_time = str(datetime.now())
        answer = faceapp.get(photo, max_num=0)
        photocopy = photo.copy()
        for opt in answer:
            em = opt['embedding']
            x1, y1, x2, y2 = opt['bbox'].astype(int)
            oruko, ipo = searchnet(dataf, feature_col, test_vector=em, name_role_cos=name_role_cos, thresh=thresh)

            print(f"Face detected: {oruko}, {ipo}, Coordinates: {x1}, {y1}, {x2}, {y2}")

            if oruko != 'Unknown':
                cv2.rectangle(photocopy, (x1, y1), (x2, y2), (40, 240, 50), 1)
                cv2.putText(photocopy, oruko, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 250, 60), 1)
                cv2.putText(photocopy, current_time, (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 150, 200), 1)

                self.logs['Name'].append(oruko)
                self.logs['Role'].append(ipo)
                self.logs['Course'].append(course)
                self.logs['current_time'].append(current_time)
            else:
                cv2.rectangle(photocopy, (x1, y1), (x2, y2), (100, 90, 240), 1)
                cv2.putText(photocopy, oruko, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (145, 50, 200), 1)

        return photocopy


### Registration Code

class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def resetVid(self):
        self.sample= 0

    def get_embeddings(self,frame):

        results = faceapp.get(frame,max_num=1)
        embeddings = None # if no face is detected
        for res in results:
            self.sample +=1
            x1,y1,x2,y2 = res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,250,20),1)
            # Put text sample info
            txt = f"Samples = {self.sample}"
            cv2.putText(frame,txt,(x1,y1),cv2.FONT_ITALIC,0.6,(30,200,255),1)

            embeddings = res['embedding']
        return frame,embeddings

    def save_data_in_redis(self,name,role):
        #Validation
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'


        # load face_embedd.txt
        xarray = np.loadtxt('face_embedding.txt',dtype=np.float32) # flatten

        #Covert to array (proper shape)
        received_samples = int(xarray.size/512)
        x_array = xarray.reshape(received_samples,512)
        x_array = np.asarray(x_array)

        #Cal mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # Push to Redis
        r.hset(name='NHS-free-db',key=key,value=x_mean_bytes)

        os.remove('face_embedding.txt')
        self.resetVid()

        return True
