import * as React from 'react';
import {useState} from 'react';
import DatePicker from 'react-native-date-picker'

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
  TextInput,
  ScrollView,
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
import SelectDropdown from 'react-native-select-dropdown';
import FontAwesome from 'react-native-vector-icons/FontAwesome';
import Ionicons from 'react-native-vector-icons/Ionicons';
import GetLocation from 'react-native-get-location'

var base64_str = "";
var base64_str2 = "";
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
            onPress={() => navigation.navigate('Rock Mass Rating (RMR)')}>
            <Text style={styles.HomeButtons}>Rock mass Rating (RMR)</Text>
        </TouchableOpacity>
        {/*<TouchableOpacity
            activeOpacity={0.5}
            onPress={() => navigation.navigate('Geological Strength Index(GSI)')}>
            <Text style={styles.HomeButtons}>Geological Strength Index(GSI)</Text>
    </TouchableOpacity>*/}
        <TouchableOpacity
            activeOpacity={0.5}
            onPress={() => navigation.navigate('Kinematic Analysis')}>
            <Text style={styles.HomeButtons}>Kinematic Analysis</Text>
        </TouchableOpacity>
        {/*<TouchableOpacity
            activeOpacity={0.5}
            onPress={() => navigation.navigate('Failure2')}>
            <Text style={styles.HomeButtons}>Detect Failure2</Text>
        </TouchableOpacity>*/}
        <TouchableOpacity
            activeOpacity={0.5}
            onPress={() => navigation.navigate('CollectData')}>
            <Text style={styles.HomeButtons}>Collect Data</Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
}

function RmrScreen({ navigation }) {

  const [filePath, setFilePath] = useState({});
  const [isCameraOrGallerySelected, setIsCameraOrGallerySelected] = useState(false);
  const [rmrInputs, setRmrInputs] = useState({imageUri: "", scale: 1, ucs: -1, jointCond: -1, groundWater: -1});
  const [rqd, setRqd] = useState(-0.1);
  const [jointSpacing, setJointSpacing] = useState('');
  const [gsi, setGsi] = useState("");
  const jointCondition = ["Very rough surfaces, Not continuous, No seperation, Unweathered wall rock", "Slightly rough surfaces, seperation < 1 mm, slightly weathered walls", "Slightly rough surfaces, Seperation < 1 mm, Highly weathered walls", "Slickensided surfaces, seperation 1-5 mm, continuous", "Seperation > 5 mm, continuous"];
  const groundWaterCondition = ["Completely dry", "Damp", "Wet", "Dripping", "Flowing"];

  
 
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
        'http://10.0.2.2:5000/result', requestOptions
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
  setGsi(dataa['gsi']);
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
      //console.log('came here');
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
        setIsCameraOrGallerySelected(true);
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
      setIsCameraOrGallerySelected(true);
      base64_str = 'data:image/jpeg;base64,' + response['assets'][0]['base64'];
      setFilePath(response['assets'][0]);
    });
  };

  return (
    <SafeAreaView style={{flex: 1}}>
      <ScrollView style={styles.inputsContainer}>
      <TouchableOpacity
          activeOpacity={0.5}
          style={{alignItems: 'center',
          backgroundColor: "gray",
          backgroundColor: '#DDDDDD',
          padding: 2,
          marginLeft:15,
          width: 160,
          fontSize:5,
          borderRadius: 10,}}
          onPress={() => alert("1. Scale should be provided as the true width of the portion captured in meters.\n\n2. UCS should be given in MPa\n\n3. RMR will not be obtained if any of the parameters are missing, but RQD and joint spacing will be obtained if an image is uploaded.\n\n4. Upload a high quality image of the rock mass. Avoid noise such as grass, roads, trees, etc. The estimation may be deviated if the image contains these noise mentioned.")}>
          <Text style={{color:'black',}}>Instructions for RMR</Text>
        </TouchableOpacity>
        <TextInput style={styles.input} placeholder="Enter Scale" keyboardType="phone-pad" onChangeText={(value) => setRmrInputs({...rmrInputs, scale: value})}/>
        <TextInput style={styles.input} placeholder="Enter Uniaxial Compressive Strength(UCS) in Mpa" keyboardType="phone-pad" onChangeText={(value) => setRmrInputs({...rmrInputs, ucs: value})}/>
        <SelectDropdown
            data={jointCondition}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              if(index == 0){
                setRmrInputs({...rmrInputs, jointCond: 30});
              }else if(index == 1){
                setRmrInputs({...rmrInputs, jointCond: 25});
              }else if(index == 2){
                setRmrInputs({...rmrInputs, jointCond: 20});
              }else if(index == 3){
                setRmrInputs({...rmrInputs, jointCond: 10});
              }else if(index == 4){
                setRmrInputs({...rmrInputs, jointCond: 0});
              }
            }}
            defaultButtonText={'Condition of joints'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyleSmall}
          />
          <SelectDropdown
            data={groundWaterCondition}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              if(index == 0){
                setRmrInputs({...rmrInputs, groundWater: 15});
              }else if(index == 1){
                setRmrInputs({...rmrInputs, groundWater: 10});
              }else if(index == 2){
                setRmrInputs({...rmrInputs, groundWater: 7});
              }else if(index == 3){
                setRmrInputs({...rmrInputs, groundWater: 4});
              }else if(index == 4){
                setRmrInputs({...rmrInputs, groundWater: 0});
              }
            }}
            defaultButtonText={'Ground water condition'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />

        {!isCameraOrGallerySelected?<Text style={styles.textStyle}>Upload or capture image to get other required parameters.</Text>:null}

        
        {!isCameraOrGallerySelected?<TouchableOpacity
          activeOpacity={0.5}
          style={{alignItems: 'center',
          backgroundColor: "gray",
          backgroundColor: '#DDDDDD',
          padding: 2,
          marginVertical: 10,
          marginLeft: 70,
          width: 250,
          borderRadius: 10,}}
          onPress={() => captureImage('photo')}>
          <Text style={styles.textStyle}>Camera</Text>
        </TouchableOpacity>:null}

        {!isCameraOrGallerySelected?<TouchableOpacity
          activeOpacity={0.5}
          style={{    alignItems: 'center',
          backgroundColor: "gray",
          backgroundColor: '#DDDDDD',
          padding: 2,
          marginVertical: 10,
          marginLeft: 70,
          width: 250,
          borderRadius: 10,}}
          onPress={() => chooseFile('photo')}>
          <Text style={styles.textStyle}>Gallery</Text>
        </TouchableOpacity>:null}

        {base64_str != ""?<Image source={{uri: filePath['uri']}} style={{width: 200, height: 200, margin: 5, alignItems: 'center', marginLeft:95}}/>:null}
        {isCameraOrGallerySelected?<TouchableOpacity
          activeOpacity={0.5}
          style={{alignItems: 'center',
          backgroundColor: "gray",
          backgroundColor: '#DDDDDD',
          padding: 2,
          marginVertical: 10,
          marginLeft: 70,
          width: 250,
          borderRadius: 10,}}
          onPress={() => afterClick()}>
          <Text style={styles.textStyle}>Process Image</Text>
        </TouchableOpacity>:null}
        <Text style={styles.textStyle}>{rqd != -0.1 ? 'Rock Quality Designation(RQD): '+rqd:null}</Text>
        <Text style={styles.textStyle}>{jointSpacing != '' ? 'Joint Spacing: '+jointSpacing: null}</Text>
        <Text style={styles.textStyle}>{gsi != '' ? 'Geological Strength Index(GSI): '+gsi: null}</Text>

        
        </ScrollView>
    </SafeAreaView>
  );
}



function KinematicAnalysisScreen({ navigation }) {
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
        'http://10.0.2.2:5000/failure', requestOptions
      );
      var data = await response.text();
      setOutput(data);
      //array = JSON.stringify(data);
      //console.log(data);
      
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

function FailureScreen2({ navigation }) {
  const [inputs, setInputs] = useState([{strike2: -1, Dip2: -1}]);
  const addHandler = ()=>{
    const _inputs = [...inputs];
    _inputs.push({key: '', value: ''});
    setInputs(_inputs);
  }

  const deleteHandler = (key)=>{
    const _inputs = inputs.filter((input,index) => index != key);
    setInputs(_inputs);
  }

  const inputHandler = (text, key)=>{
    const _inputs = [...inputs];
    _inputs[key].value = text;
    _inputs[key].key   = key;
    setInputs(_inputs);
    
  }

  return (
    <View style={styles.container}>
      <ScrollView style={styles.inputsContainer}>
      {inputs.map((input, key)=>(
        <View style={styles.inputContainer} key={key}>
          <TextInput placeholder={"Enter Strike"} value={input.value}  onChangeText={(text)=>inputHandler(text,key)}/>
          <TouchableOpacity onPress = {()=> deleteHandler(key)}>
            <Text style={{color: "red", fontSize: 13}}>Delete</Text>
          </TouchableOpacity> 
        </View>
      ))}
      </ScrollView>
      <Button title="Add" onPress={addHandler} />
    </View>
  );

  
}

function DataScreen({ navigation }) {
  const [lslide, setLslide] = useState({lName: "", latitude: 0.0, longitude:0.0, cLocation: "", shape: "", dateEvent: "", dateRecord: "", material: "", movement: "", luf: "", damage: "", triggerr: "", isReactivated: "", pActive: "", pReactive: "", hDegree: "", pEvolution: ""});
  const shapes = ["Point", "Line", "Polygon"];
  const typeOfMat = ["Debris", "Earth", "Rock"];
  const typeOfMov = ["Slide", "Flow", "Fall", "Rotational Slump", "Flow slide"];
  const triggerFactor = ["Rainfall", "Earthquake", "Human activity", "Others"];
  const reac = ["Yes", "No"];
  const presentlyActive = ["Yes", "No"];
  const possibleReac = ["Yes", "No"];
  const hazardDegree = ["No Hazard", "Low", "Medium", "High"];
  const possibleEvol = ["Up", "Down", "Widening"];
  const optionsForLocation = ["Automatically", "Manually"];
  const [date1, setDate1] = useState(new Date());
  const [date2, setDate2] = useState(new Date());
  const [open1, setOpen1] = useState(false);
  const [open2, setOpen2] = useState(false);
  const [eventDate, setEventDate] = useState('Date of Event');
  const [locationOption, setLocationOption] = useState(0);
  const [lat, setLat] = useState(0.0);
  const [longi, setLongi] = useState(0.0);

  const afterClick = async () => {
    console.log(JSON.stringify(lslide));

    // if(Ds == null || Dd == null || Ss == null || Sd == null || Fa == null){
    //   alert('Some fields are empty! All fields are mandatory.');
    //   return;
    // }
    const requestOptions = {
      method: 'POST',
      body: JSON.stringify(lslide),
      headers: {
        'Content-Type': 'application/json'
      },
    };

  const sendToDb = async () => {
    //var array;
    try {
      const response = await fetch(
        'http://10.0.2.2:80/db_api/insert.php', requestOptions
      );
      var data = await response.json();
      //setOutput(data);
      //array = JSON.stringify(data);
      console.log(data);
      alert('Data submitted successfully');
      
    } catch (error) {
      console.error(error);
      alert('Cannot process')
      return;
    }
  };

  await sendToDb();



  };

  return (
    <SafeAreaView style={{flex:1}}>
      <ScrollView style={styles.inputsContainer}>
      <TextInput style={styles.input} placeholder="Landslide Name" onChangeText={(value) => setLslide({...lslide, lName: value})}/>
      <SelectDropdown
            data={optionsForLocation}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              if(selectedItem == "Manually"){
                setLocationOption(2);
              }else{
                setLocationOption(1);
                GetLocation.getCurrentPosition({
                  enableHighAccuracy: true,
                  timeout: 15000,
              })
              .then(location => {
                  setLat(location['latitude']);
                  setLongi(location['longitude']);
                  setLslide({...lslide, latitude: location['latitude']});
                  setLslide({...lslide, longitude: location['longitude']});
              })
              .catch(error => {
                  const { code, message } = error;
                  console.warn(code, message);
              })
              }
            }}
            defaultButtonText={'How do you want to set location?'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
          {locationOption==2 ? <TextInput style={styles.input} placeholder="Enter location" onChangeText={(value) => setLslide({...lslide, cLocation: value})}/> : null}
          {locationOption==1? <Text style={styles.textStyle}>Latitude = {lat},  Longitude = {longi}</Text>: null}
      <TouchableOpacity
          activeOpacity={0.5}
          style={styles.input}
          onPress={() => setOpen1(true)}>
          <Text>{lslide['dateEvent'] == "" ? 'Date of Event' : lslide['dateEvent']}</Text>
      </TouchableOpacity>
      <DatePicker
        modal
        mode='date'
        open={open1}
        date={date1}
        onConfirm={(date) => {
          setOpen1(false)
          setDate1(date)
          var s = JSON.stringify(date).slice(1,11);
          setLslide({...lslide, dateEvent: s})
          //console.log(s);
        }}
        onCancel={() => {
          setOpen1(false)
        }}
      />
      <TouchableOpacity
          activeOpacity={0.5}
          style={styles.input}
          onPress={() => setOpen2(true)}>
          <Text>{lslide['dateRecord'] == "" ? 'Date of Record' : lslide['dateRecord']}</Text>
      </TouchableOpacity>
      <DatePicker
        modal
        mode='date'
        open={open2}
        date={date2}
        onConfirm={(date) => {
          setOpen2(false)
          setDate2(date)
          var s = JSON.stringify(date).slice(1,11);
          setLslide({...lslide, dateRecord: s})
          //console.log(s);
        }}
        onCancel={() => {
          setOpen2(false)
        }}
      />
      <SelectDropdown
            data={typeOfMat}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              setLslide({...lslide, material: selectedItem})
            }}
            defaultButtonText={'Type of material'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
          <SelectDropdown
            data={typeOfMov}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              setLslide({...lslide, movement: selectedItem})
            }}
            defaultButtonText={'Type of Movement'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
        <TextInput style={styles.input} placeholder="Land-use Features (eg: Forest, Road, River, Agriculture field, House, etc..)" onChangeText={(value) => setLslide({...lslide, luf: value})}/>
        <TextInput style={styles.input} placeholder="Damage (eg: Road, House, School, Forest, Communication line, etc..)" onChangeText={(value) => setLslide({...lslide, damage: value})}/>
        <SelectDropdown
            data={triggerFactor}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              setLslide({...lslide, triggerr: selectedItem})
            }}
            defaultButtonText={'Trigger factor'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
          <SelectDropdown
            data={reac}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              setLslide({...lslide, isReactivated: selectedItem})
            }}
            defaultButtonText={'Reactivated?'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
          <SelectDropdown
            data={presentlyActive}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              setLslide({...lslide, pActive: selectedItem})
            }}
            defaultButtonText={'Presently active?'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
          <SelectDropdown
            data={possibleReac}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              setLslide({...lslide, pReactive: selectedItem})
            }}
            defaultButtonText={'Possible Reactivation'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
          <SelectDropdown
            data={hazardDegree}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              setLslide({...lslide, hDegree: selectedItem})
            }}
            defaultButtonText={'Hazard Degree'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
          <SelectDropdown
            data={possibleEvol}
            // defaultValueByIndex={1}
            // defaultValue={'Egypt'}
            onSelect={(selectedItem, index) => {
              //console.log(selectedItem, index);
              setLslide({...lslide, pEvolution: selectedItem})
            }}
            defaultButtonText={'Possible Evolution'}
            buttonTextAfterSelection={(selectedItem, index) => {
              return selectedItem;
            }}
            rowTextForSelection={(item, index) => {
              return item;
            }}
            buttonStyle={styles.dropdown1BtnStyle}
            buttonTextStyle={styles.dropdown1BtnTxtStyle}
            renderDropdownIcon={isOpened => {
              return <FontAwesome name={isOpened ? 'chevron-up' : 'chevron-down'} color={'#444'} size={18} />;
            }}
            dropdownIconPosition={'right'}
            dropdownStyle={styles.dropdown1DropdownStyle}
            rowStyle={styles.dropdown1RowStyle}
            rowTextStyle={styles.dropdown1RowTxtStyle}
          />
      <TouchableOpacity
          activeOpacity={0.5}
          style={{    alignItems: 'center',
          backgroundColor: "gray",
          backgroundColor: '#DDDDDD',
          padding: 2,
          marginVertical: 10,
          marginLeft: 70,
          width: 250,
          borderRadius: 10,}}
          onPress={() => afterClick()}>
          <Text style={styles.textStyle}>Submit Data</Text>
      </TouchableOpacity>
      </ScrollView>
      {/*<Button title="Add" onPress={} />*/}
    </SafeAreaView>
  );
}

const Stack = createStackNavigator();

function MyStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={HomeScreen} options={{headerShown: false}}/>
      <Stack.Screen name="Rock Mass Rating (RMR)" component={RmrScreen} />
      {/*<Stack.Screen name="Geological Strength Index(GSI)" component={GSIscreen} />*/}
      <Stack.Screen name="Kinematic Analysis" component={KinematicAnalysisScreen} />
      {/*<Stack.Screen name="Failure2" component={FailureScreen2} />*/}
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
    width: "93%",
    borderWidth: 1,
    borderRadius: 10,
    margin: 15,
    padding: 10,
  },
  inputsContainer: {
    flex: 1, marginBottom: 20,
    backgroundColor: '#fff',
  },
  inputContainer: {
    backgroundColor: '#fff',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: "lightgray"
  },
  dropdown1BtnStyle: {
    margin: 15,
    padding: 2,
    width: '93%',
    height: 48,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderRadius: 10,
    borderColor: "gray",
    //alignItems: 'center',
  },
  dropdown1BtnTxtStyle: {color: 'gray', textAlign: 'left', fontSize:14},
  dropdown1DropdownStyle: {backgroundColor: '#EFEFEF'},
  dropdown1RowStyle: {backgroundColor: '#EFEFEF', borderBottomColor: '#C5C5C5'},
  dropdown1RowTxtStyle: {color: '#444', textAlign: 'left'},
  dropdown1RowTxtStyleSmall: {fontSize:10,color: '#444', textAlign: 'left'},
});
