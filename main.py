from flask import Flask, request
import numpy as np
import joblib

app = Flask(__name__)
@app.route('/predict_price', methods=['GET'])
def print_hi():
    model = joblib.load('model.pkl')
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')
    args = request.args
    floor=args.get("floor", default=np.nan, type=int)
    open_plan=args.get("open_plan", default=np.nan, type=int)
    rooms = args.get("rooms", default=np.nan, type=int)
    studio = args.get("studio", default=np.nan, type=int)
    area = args.get("area", default=np.nan, type=float)
    kitchen_area = args.get("kitchen_area", default=np.nan, type=float)
    living_area = args.get("living_area", default=np.nan, type=float)
    agent_fee = args.get("agent_fee", default=np.nan, type=float)
    renovation = args.get("renovation", default=np.nan, type=int)

    row = [
        floor,
        open_plan,
        rooms,
        studio,
        area,
        kitchen_area,
        living_area,
        agent_fee,
        renovation,
    ]
    row = np.array(row).reshape((1,-1))
    area_per_room = row[0][4] / row[0][2]
    living_area_ratio = row[0][6] / row[0][4]
    kitchen_area_ratio = row[0][5] / row[0][6]
    additional_features = [
        area_per_room,
        kitchen_area_ratio,
        living_area_ratio,
    ]
    additional_features = np.array(additional_features)
    row = np.concatenate([row[0], additional_features]).reshape((1,-1))
    row = imputer.transform(row)
    row = scaler.transform(row)
    predict = model.predict(row)

    return str(predict[0])

if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')

