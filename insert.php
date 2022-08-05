<?php
    $CN = mysqli_connect("localhost", "admin", "admin");
    $DB = mysqli_select_db($CN, "landslide_data");

    $EncodedData = file_get_contents('php://input');
    $DecodedData = json_decode($EncodedData, true);

    $lName = $DecodedData['lName'];
    $latitude = $DecodedData['latitude'];
    $longitude =$DecodedData['longitude'];
    $cLocation = $DecodedData['cLocation'];
    $dateEvent = $DecodedData['dateEvent'];
    $dateRecord = $DecodedData['dateRecord'];
    $material = $DecodedData['material'];
    $movement = $DecodedData['movement'];
    $luf = $DecodedData['luf']; 
    $damage = $DecodedData['damage'];
    $triggerr = $DecodedData['triggerr'];
    $isReactivated = $DecodedData['isReactivated'];
    $pActive = $DecodedData['pActive'];
    $pReactive = $DecodedData['pReactive'];
    $hDegree = $DecodedData['hDegree'];
    $pEvolution = $DecodedData['pEvolution'];

    $IQ = "insert into lslide(lName,latitude,longitude,cLocation,dateEvent,dateRecord,material,movement,luf,damage,triggerr,isReactivated,pActive,pReactive,hDegree,pEvolution) values('$lName',$latitude,$longitude,'$cLocation','$dateEvent','$dateRecord','$material','$movement','$luf','$damage','$triggerr','$isReactivated','$pActive','$pReactive','$hDegree','$pEvolution')";

    $R = mysqli_query($CN, $IQ);

    if($R)
    {
        $Message = "Data submitted successfully";
    }
    else{
        $Message = "Error!!!";
    }
    $Response[] = array("Message"=>$Message);
    echo json_encode($Response);

?>