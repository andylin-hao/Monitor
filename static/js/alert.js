class Connection {
    constructor(){
        this.xhr = new XMLHttpRequest();
        this.stringLength = 0;
        this.xhr.open('GET', "http://127.0.0.1:8000/monitor/alert/");
        this.xhr.onreadystatechange = this.stateChange();
        this.xhr.send('connect');
    }

    stateChange(){
        let _this = this;
        function f() {
            if (_this.xhr.readyState === 3) {
                let end = _this.xhr.responseText.length;
                let data = _this.xhr.responseText.slice(_this.stringLength, end);
                console.log(data);

                if(data.length !== 0){
                    data = JSON.parse(data);
                    console.log(data);
                    let alerts = data['data'];
                    let index = data['index'];
                    let username = data['username'];
                    for(let i = 0; i < alerts.length; i++){
                        let div = document.createElement("div");
                        div.setAttribute("class", "msg");
                        div.innerHTML =
                            `<h4>`+ username +  ` became ` + alerts[i] + `</h4>
                             <form action="/monitor/alertPage/" method="post">
                                <input type="hidden" name="index" value="` + String(index) + `">
                                <input type="hidden" name="pos" value="` + String(i) +`">
                                <button class="mui-btn mui-btn--fab mui-btn--danger delete"></button>
                             </form>`;
                        document.querySelectorAll("#alert")[0].appendChild(div)
                    }
                }


                _this.stringLength = end;
                if (_this.xhr.responseText.length > 1024 * 1024 * 200) {
                    _this.autoProcess();
                }
            }
        }
        return f;
    }

    autoProcess(){
        this.xhr.abort();
        this.xhr = new XMLHttpRequest();
        this.stringLength = 0;
        this.xhr.open('GET', "http://127.0.0.1:8000/monitor/alert/");
        this.xhr.onreadystatechange = this.stateChange();
        this.xhr.send('connect');
    }
}

function submitCheck(){
    let input = parseFloat($("#threshold")[0].value);
    if(input < 0 || input > 1){
        window.alert("Please input threshold between 0 and 1");
    }
    else{
        $("#thresForm").submit();
    }
}

let connection = new Connection();