// 'use strict';
//
// let videoElement = document.querySelector('video');
// let canvasElement = document.createElement('canvas');
// let audioDevices = [];
// let videoDevices = [];
//
// navigator.mediaDevices.enumerateDevices()
//     .then(gotDevices).then(getStream).catch(handleError);
//
// sendRequest();
//
// function gotDevices(deviceInfos) {
//     for (let i = 0; i !== deviceInfos.length; ++i) {
//         let deviceInfo = deviceInfos[i];
//         if (deviceInfo.kind === 'audioinput') {
//             audioDevices.push(deviceInfo.deviceId);
//         } else if (deviceInfo.kind === 'videoinput') {
//             videoDevices.push(deviceInfo.deviceId);
//         } else {
//             console.log('Found one other kind of source/device: ', deviceInfo);
//         }
//     }
// }
//
// function getStream() {
//     if (window.stream) {
//         window.stream.getTracks().forEach(function (track) {
//             track.stop();
//         });
//     }
//
//     let constraints = {
//         audio: {
//             deviceId: {exact: audioDevices[0].value}
//         },
//         video: {
//             deviceId: {exact: videoDevices[0].value}
//         }
//     };
//
//     navigator.mediaDevices.getUserMedia(constraints).then(gotStream).catch(handleError);
// }
//
// function gotStream(stream) {
//     window.stream = stream; // make stream available to console
//     videoElement.srcObject = stream;
// }
//
// function handleError(error) {
//     console.log('Error: ', error);
// }

function sendRequest() {
    $.ajax({
        data: {
            'csrfmiddlewaretoken': $('input[name="csrfmiddlewaretoken"]').attr("value"),
        },
        type: "post",
        url: "/monitor/registerFace/",
        success: function (res) {
            let data = JSON.parse(res);
            $("img").attr("src", "data:image/jpeg;base64," + data['img']);
            progress_bar.set_progress(parseFloat(data['progress']).toFixed(2)*100);
            sendRequest();
            if(data['complete'] === 'true')
                window.location.href = 'http://127.0.0.1:8000/monitor/'
        },
        fail: function (res) {
            console.log("error");
            console.log(res);
        }
    })
}

let progress_bar = new ProgressBar({
    'type': 'bar',
    'container': '#progress-bar-bar',
    'background': '#FFFFFF',
    'foreground': '#03DAC5',
    'complete': () => {
    }
});
sendRequest();
