from flask import Flask, json, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from OpenSSL import SSL
import datetime as dt
import logging, ssl
import baseFunction

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)
api = Api(app)

# iOS앱에서 사용자 uid 및 사진 번호 Server로 POST
@app.route('/Auth', methods = ['POST'])
def Auth():
    jsonData = request.get_json()
    uid = jsonData["uid"] # 사용자 uid
    imageNum = int(jsonData["numbers"]) # 사진 번호
    isCavity = jsonData["isCavity"] # 충치 모델인지 치주 모델인지 선택하는 파라미터
    datetime = baseFunction.todayDate() # 업로드된 날짜
    checkResults = baseFunction.downloadImage(uid, imageNum, isCavity) # Firebase Storage에서 이미지 다운로드
    if checkResults:
        return {"status": "OK", "datetime": f"{datetime}"}
    else :
        return {"status": "Fail", "datetime": f"{datetime}"}

# iOS앱에서 사용자 uid 및 사진 번호 Server로 POST
@app.route('/Charts', methods = ['POST'])
def Charts():
    jsonData = request.get_json()
    uid = jsonData["uid"] # 사용자 uid
    datetime = baseFunction.todayDate() # 업로드된 날짜
    return baseFunction.makeChartsinDatabase(uid)

if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    cert = '/workspace/Cert/cert.pem'
    pkey = '/workspace/Cert/privkey.pem'
    chainkey = '/workspace/Cert/fullca.pem'
    context.load_verify_locations('/workspace/Cert/fullca.pem')
    context.load_cert_chain(cert, pkey)
    app.run(debug=False, host='192.168.10.6', port=6000, ssl_context=context)