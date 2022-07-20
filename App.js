import * as React from 'react';
import {useState} from 'react';

import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Platform,
  PermissionsAndroid,
  Button,
  ImageBackground,
  TextInput
} from 'react-native';
 
import {
  launchCamera,
  launchImageLibrary
} from 'react-native-image-picker';

import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { Modal } from 'react-native';
import ImageViewer from 'react-native-image-zoom-viewer';
import { out } from 'react-native/Libraries/Animated/Easing';

var base64_str = "";
var dataa;

function HomeScreen({ navigation }) {
  return (
    <ImageBackground
    source={require('./bg_img4.jpg')}
    style={styles.background}
    >
      
      {/*<Text style={styles.HometitleText}>
        What do you want to calculate?
      </Text>*/}
      <View>
        <TouchableOpacity
            activeOpacity={0.5}
            onPress={() => navigation.navigate('Rock mass characterisation')}>
            <Text style={styles.HomeButtons}>Rock mass characterisation</Text>
        </TouchableOpacity>
        <TouchableOpacity
            activeOpacity={0.5}
            onPress={() => navigation.navigate('Geological Strength Index(GSI)')}>
            <Text style={styles.HomeButtons}>Geological Strength Index(GSI)</Text>
        </TouchableOpacity>
        <TouchableOpacity
            activeOpacity={0.5}
            onPress={() => navigation.navigate('Failure')}>
            <Text style={styles.HomeButtons}>Detect Failure</Text>
        </TouchableOpacity>
        <TouchableOpacity
            activeOpacity={0.5}
            onPress={() => navigation.navigate('CollectData')}>
            <Text style={styles.HomeButtons}>Collect Data</Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
}

function RockScreen({ navigation }) {

  const [filePath, setFilePath] = useState({});
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
  var dict = {
    uri: p_str
  };
  setFilePath(dict);
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
      console.log('came here');
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
      <View style={styles.container}>

      <TouchableOpacity
          activeOpacity={0.5}
          style={styles.imageStyle}
          onPress={() => navigation.navigate('GSI')}>
          <Image
          source={{uri: filePath['uri']}}
          style={styles.imageStyle}
        />
        </TouchableOpacity>
        
        <Text style={styles.textStyle}>{rqd != -0.1 ? 'RQD(Rock Quality Designation): '+rqd:null}</Text>
        <Text style={styles.textStyle}>{jointSpacing != '' ? 'Joint Spacing: '+jointSpacing: null}</Text>
        
        <TouchableOpacity
          activeOpacity={0.5}
          style={styles.buttonStyle}
          onPress={() => captureImage('photo')}>
          <Text style={styles.textStyle}>Camera</Text>
        </TouchableOpacity>

        <TouchableOpacity
          activeOpacity={0.5}
          style={styles.buttonStyle}
          onPress={() => chooseFile('photo')}>
          <Text style={styles.textStyle}>Gallery</Text>
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
}

function GSIscreen({ navigation }) {
  return (
    <SafeAreaView style={{flex: 1}}>
      <View style={styles.container}>
        
        {/*<Image
          source={{uri: filePath['uri']}}
          style={styles.imageStyle}
  />*/}
        
        <TouchableOpacity
          activeOpacity={0.5}
          style={styles.buttonStyle}
          onPress={() => chooseFile('photo')}>
          <Text style={styles.textStyle}>calculate GSI</Text>
        </TouchableOpacity>
        
      </View>
    </SafeAreaView>
  );
}

function FailureScreen({ navigation }) {
  const [Ds, setDs] = useState();
  const [Dd, setDd] = useState();
  const [Ss, setSs] = useState();
  const [Sd, setSd] = useState();
  const [Fa, setFa] = useState();
  const [output, setOutput] = useState("");

  const afterClick = async () => {
    if(Ds == null || Dd == null || Ss == null || Sd == null || Fa == null){
      alert('Some fields are empty! All fields are mandatory.');
      return;
    }
    const requestOptions = {
      method: 'POST',
      body: Ds.toString() + " " + Ss.toString() + " " + Dd.toString() + " " + Sd.toString() + " " + Fa.toString(),
    };

  const processFailure = async () => {
    //var array;
    try {
      const response = await fetch(
        'https://landslides-btp.herokuapp.com/failure', requestOptions
      );
      var data = await response.text();
      setOutput(data);
      //array = JSON.stringify(data);
      console.log(data);
      
    } catch (error) {
      console.error(error);
      alert('Cannot process')
      return;
    }
  };

  await processFailure();



  };

  return (
    <SafeAreaView style={{flex: 1}}>
      <View style={styles.container}>
      <TextInput style={styles.input} placeholder="Discontinuity strike" keyboardType="phone-pad" onChangeText={(value) => setDs(value)}/>
      <TextInput style={styles.input} placeholder="Discontinuity Dip" keyboardType="phone-pad" onChangeText={(value) => setDd(value)}/>
      <TextInput style={styles.input} placeholder="Slope strike" keyboardType="phone-pad" onChangeText={(value) => setSs(value)}/>
      <TextInput style={styles.input} placeholder="Slope Dip" keyboardType="phone-pad" onChangeText={(value) => setSd(value)}/>
      <TextInput style={styles.input} placeholder="Friction angle" keyboardType="phone-pad" onChangeText={(value) => setFa(value)}/>
      <Text style={styles.failureOutput}>{output}</Text>
      <TouchableOpacity
          activeOpacity={0.5}
          style={styles.buttonStyle}
          onPress={() => afterClick()}>
          <Text style={styles.textStyle}>Process</Text>
      </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

function DataScreen({ navigation }) {
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      <Button title="Go back" onPress={() => navigation.goBack()} />
    </View>
  );
}

const Stack = createStackNavigator();

function MyStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={HomeScreen} options={{headerShown: false}}/>
      <Stack.Screen name="Rock mass characterisation" component={RockScreen} />
      <Stack.Screen name="Geological Strength Index(GSI)" component={GSIscreen} />
      <Stack.Screen name="Failure" component={FailureScreen} />
      <Stack.Screen name="CollectData" component={DataScreen} />
    </Stack.Navigator>
  );
}

export default function App() {
  return (
    <NavigationContainer>
      <MyStack />
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  background: {
    width: '100%',
    height: '100%'
  },
  HometitleText: {
    fontSize: 26,
    fontWeight: 'bold',
    textAlign: 'center',
    paddingVertical: 20,
    color: 'white',
  },
  container: {
    flex: 1,    fontWeight: 'bold',
    textAlign: 'center',
    paddingVertical: 20,
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
    margin: 5,
    color: 'black',
    textAlign: 'center',
  },
  failureOutput: {
    padding: 5,
    margin: 5,
    color: 'black',
    fontWeight: 'bold',
    fontSize: 13,
  },
  buttonStyle: {
    alignItems: 'center',
    backgroundColor: "gray",
    backgroundColor: '#DDDDDD',
    padding: 2,
    marginVertical: 10,
    width: 250,
    borderRadius: 10,
  },
  HomeButtons: {
    backgroundColor: '#dcdada',
    color: 'black',
    width: "75%",
    borderRadius: 15,
    textAlign: 'center',
    marginLeft: '11%',
    padding: "2%",
    fontSize:  22,
    marginTop: '23%'

  },
  imageStyle: {
    width: 200,
    height: 200,
    margin: 5,
  },
  input: {
    borderColor: "gray",
    width: "100%",
    borderWidth: 1,
    borderRadius: 10,
    margin: 15,
    padding: 10,
  },
});
