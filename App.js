import { Audio, RecordingOptionsPresets } from "expo-av";
import React, { useState } from 'react';
import { StatusBar, StyleSheet, Text, TouchableOpacity, View, ImageBackground, Modal, TouchableWithoutFeedback } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { Platform, Image} from 'react-native';
// import { View, Image } from 'react-native';
// import DocumentPicker from 'react-native-document-picker';

const HANYANG_BLUE = '#003DA5'; // 한양대학교 로고 색상

export default function App() {
  const [result, setResult] = useState(null);
  const [showMessage, setShowMessage] = useState(false); // 문구를 보여줄지 여부를 관리하는 state
  const [modalVisible, setModalVisible] = useState(false); // 모달 창의 표시 여부를 관리하는 state
  const [isProcessing, setIsProcessing] = useState(false);  // 추가된 부분
  
  const sendAudio = async (uri) => {
    try {
      console.log("Preparing to send audio...");

      let formData = new FormData();
      console.log('formData OK');
      let uriParts = uri.split('.');
      console.log('uriParts OK');
      let fileType = uriParts[uriParts.length - 1];
      console.log('fileType OK');

      formData.append('audio', {
        uri: Platform.OS === 'ios' ? uri.replace('file://', '') : uri,
        name: `audio.${fileType}`,
        type: `audio/${fileType}`,
      });

      console.log("FormData created:", formData);

      let response = await fetch('http://172.17.75.62:5000/process_audio', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log("Response received:", response);

      if (response.ok) {
        let data = await response.json();
        console.log('Server response:', data);
        setResult(data); // 서버 응답 결과 저장
      } else {
        console.error('Server response error:', response.status);
        setResult(null);
      }
    } catch (error) {
      console.error('Server communication error:', error);
      setResult(null);
    } finally {
      setIsProcessing(false); // 파일 전송 후 다시 초기 상태로 설정
    }
}

  const pickFile = async () => {
    try {
      console.log("Trying to pick a file...");
      let file = await DocumentPicker.getDocumentAsync({
        type: 'audio/*',
      });
      console.log("File picked:", file);
      console.log('File type:', file.type);
      file.type = 'success'

      if (!file.canceled) {
        console.log('File picked successfully:');
        const fileInfo = file.assets[0]; // 파일 정보에 접근

        console.log('URI:', fileInfo.uri);
        console.log('Name:', fileInfo.name); // 선택된 파일의 이름
        console.log('Type:', fileInfo.mimeType); // 선택된 파일의 MIME 타입
        console.log('Size:', fileInfo.size); // 선택된 파일의 크기
        
        setIsProcessing(true);
        sendAudio(fileInfo.uri); // 선택된 파일을 서버로 전송           
        console.log('successfully sent')
      } else console.log("File picking failed.");

    } catch (error) {
      console.error('파일 선택 중 오류 발생:', error);
    }
  }

  return (
    <ImageBackground 
      source={require('./assets/images/background.png')} 
      style={styles.container}
      resizeMode="cover" // 배경 이미지가 전체를 커버하도록 설정
    >
      <TouchableOpacity style={styles.infoButton} onPress={() => setModalVisible(true)}>
        <View style={styles.infoCircle}>
          <Text style={styles.infoText}>!</Text>
        </View>
      </TouchableOpacity>

      <View style={styles.circle}>
        <TouchableOpacity onPress={pickFile} disabled={isProcessing}>
          <Text style={styles.uploadText}>
            {isProcessing ? "음성 분석 중..." : "음성 파일을 선택하세요"}
          </Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.headerText}>딥보이스 판별</Text>
      {result && (
        <Text style={styles.resultText}>Result: {result.result}</Text>
      )}

      <Modal
        animationType="fade"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => {
          setModalVisible(!modalVisible);
        }}
      >
        <View style={styles.centeredView}>
          <TouchableWithoutFeedback onPress={() => setModalVisible(false)}>
            <View style={styles.modalOverlay} />
          </TouchableWithoutFeedback>
          <View style={styles.modalView}>
            <Text style={styles.modalText}>전자공학부 캡스톤디자인 Deepfaker 팀 입니다. {"\n"} AI voice와 Real voice를 딥러닝을 활용해 두가지로 분류하는 어플리케이션입니다.</Text>
            <TouchableOpacity
              style={styles.closeButton}
              onPress={() => setModalVisible(false)}
            >
              <Text style={styles.closeButtonText}>X</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <StatusBar style="auto" />
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%', // 전체 너비 사용
    height: '100%' // 전체 높이 사용
  },
  infoButton: {
    position: 'absolute',
    top: 20,
    right: 20,
    zIndex: 1, // Ensure it's above other components
  },
  infoCircle: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: 'rgba(255, 255, 255, 0.8)', // 반투명 흰색 배경
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#000' // 검은색 텍스트
  },
  headerText: {
    fontSize: 36,
    fontWeight: 'bold',
    marginTop: 100,
    color: '#FFF' // 흰색 텍스트
  },
  circle: {
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: 'rgba(255, 255, 255, 0.8)', // 반투명 흰색 배경
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 0
  },
  uploadText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#000' // 검은색 텍스트
  },
  resultText: {
    fontSize: 30,
    marginTop: 20,
    color: 'red' // 파란색 텍스트
  },
  centeredView: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  modalView: {
    margin: 20,
    backgroundColor: "white",
    borderRadius: 20,
    padding: 35,
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2
    },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5
  },
  modalText: {
    marginBottom: 15,
    textAlign: "center"
  },
  modalOverlay: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0,0,0,0.5)"
  },
  closeButton: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: HANYANG_BLUE,
    borderRadius: 15,
    width: 30,
    height: 30,
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  }
});

