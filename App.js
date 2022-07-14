// Example of Image Picker in React Native
// https://aboutreact.com/example-of-image-picker-in-react-native/
 
// Import React
import React, {useState} from 'react';
// Import required components
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Platform,
  PermissionsAndroid,
} from 'react-native';
 
// Import Image Picker
// import ImagePicker from 'react-native-image-picker';
import {
  launchCamera,
  launchImageLibrary
} from 'react-native-image-picker';



var base64_str = "";
var dataa;
 
const App = () => {
  const [filePath, setFilePath] = useState({});
  const [processedImage, setProcessedImage] = useState('');
  const [button1, setButton1] = useState(true);
  const [button2, setButton2] = useState(true);
  const [rqd, setRqd] = useState(-0.1);
  const [jointSpacing, setJointSpacing] = useState('');
 
  const requestCameraPermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.CAMERA,
          {
            title: 'Camera Permission',
            message: 'App needs camera permission',
          },
        );
        // If CAMERA Permission is granted
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (err) {
        console.warn(err);
        return false;
      }
    } else return true;
  };
 
  const requestExternalWritePermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
          {
            title: 'External Storage Write Permission',
            message: 'App needs write permission',
          },
        );
        // If WRITE_EXTERNAL_STORAGE Permission is granted
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (err) {
        console.warn(err);
        alert('Write permission err', err);
      }
      return false;
    } else return true;
  };

  const afterClick = async () => {
    if(base64_str == ''){
      alert('Please input an image');
      return;
    }
    const requestOptions = {
      method: 'POST',
      body: base64_str,
      headers:{
        'Accept': 'application/json',
        'Content-Type':'apapi response in react nativeplication/json'
    }
  };

  const processImage = async () => {
    //var array;
    try {
      const response = await fetch(
        'https://landslides-btp.herokuapp.com/result', requestOptions
      );
      var data = await response.json();
      //array = JSON.stringify(data);
      dataa = data;
      console.log(dataa['linespacing']);
      
    } catch (error) {
      console.error(error);
      alert('Cannot process Image')
      return;
    }
  };

  await processImage();
  var p_str = "data:image/jpeg;base64," + dataa['res_uri'];
  setProcessedImage(p_str);
  var num = dataa['rqd'];
  setRqd(num.toFixed(4));
  var jsp_array = dataa['linespacing'];
  var jsp_str = "[";
  for(var i = 0; i < jsp_array.length; i++){
    var num = jsp_array[i];
    num = num.toFixed(4);
    var temp = num.toString();
    if(i == jsp_array.length-1){
      jsp_str = jsp_str + temp;
    }else{
      jsp_str = jsp_str + temp + ", ";
    }
  }
  jsp_str = jsp_str + ']';
  setJointSpacing(jsp_str);
  //console.log(jsp_str);
  //alert(p_str);

  };
 
  const captureImage = async (type) => {
    let options = {
      mediaType: type,
      maxWidth: 300,
      maxHeight: 550,
      quality: 1,
      videoQuality: 'low',
      durationLimit: 30, //Video max duration in seconds
      saveToPhotos: true,
      includeBase64: true,
    };
    let isCameraPermitted = await requestCameraPermission();
    let isStoragePermitted = await requestExternalWritePermission();
    if (isCameraPermitted && isStoragePermitted) {
      launchCamera(options, (response) => {
        //console.log('Response = ', response);
 
        if (response.didCancel) {
          alert('You cancelled the operation');
          return;
        } else if (response.errorCode == 'camera_unavailable') {
          alert('Camera not available on device');
          return;
        } else if (response.errorCode == 'permission') {
          alert('Permission not satisfied');
          return;
        } else if (response.errorCode == 'others') {
          alert(response.errorMessage);
          return;
        }
        base64_str = 'data:image/jpeg;base64,'+ response['assets'][0]['base64'];
        setFilePath(response['assets'][0]);
      });
    }
  };


 
  const chooseFile = (type) => {
    setButton1(false);
    let options = {
      mediaType: type,
      maxWidth: 300,
      maxHeight: 550,
      quality: 1,
      includeBase64: true,
    };
    launchImageLibrary(options, (response) => {
      //console.log('Response = ', response);
 
      if (response.didCancel) {
        alert('You cancelled the operation');
        return;
      } else if (response.errorCode == 'camera_unavailable') {
        alert('Camera not available on device');
        return;
      } else if (response.errorCode == 'permission') {
        alert('Permission not satisfied');
        return;
      } else if (response.errorCode == 'others') {
        alert(response.errorMessage);
        return;
      }
      base64_str = 'data:image/jpeg;base64,' + response['assets'][0]['base64'];
      setFilePath(response['assets'][0]);
    });
  };
 
  return (
    <SafeAreaView style={{flex: 1}}>
      <Text style={styles.titleText}>
        Rock mass characterisation
      </Text>
      <View style={styles.container}>
        
        <Image
          source={{uri: filePath['uri']}}
          style={styles.imageStyle}
        />
        {<Image
           source={processedImage ? {uri: processedImage} : null}
          style={styles.imageStyle}
        /> }

        <Text style={styles.textStyle}>{rqd != -0.1 ? 'RQD(Rock Quality Designation): '+rqd:null}</Text>
        <Text style={styles.textStyle}>{jointSpacing != '' ? 'Joint Spacing: '+jointSpacing: null}</Text>
        
        
        <TouchableOpacity
          activeOpacity={0.5}
          style={styles.buttonStyle}
          onPress={() => chooseFile('photo')}>
          <Text style={styles.textStyle}>Choose Image</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          activeOpacity={0.5}
          style={styles.buttonStyle}
          onPress={() => afterClick()}>
          <Text style={styles.textStyle}>Process Image</Text>
        </TouchableOpacity>
        
      </View>
    </SafeAreaView>
  );
};
 
export default App;
 
const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 10,
    backgroundColor: '#fff',
    alignItems: 'center',
  },
  titleText: {
    fontSize: 26,
    fontWeight: 'bold',
    textAlign: 'center',
    paddingVertical: 20,
  },
  textStyle: {
    padding: 5,
    color: 'black',
    textAlign: 'center',
  },
  buttonStyle: {
    alignItems: 'center',
    backgroundColor: '#DDDDDD',
    padding: 5,
    marginVertical: 10,
    width: 250,
  },
  imageStyle: {
    width: 200,
    height: 200,
    margin: 5,
  },
});