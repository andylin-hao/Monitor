function submitCheck() {
    let list = $("input");
    let complete = true;
    let password_1, password_2;
    list.each(function(index, element){
        if (!element.value)
            complete = false;
        if (element.name === "password_1")
            password_1 = element.value;
        if (element.name === "password_2")
            password_2 = element.value;
    });
    if(!complete){
        window.alert("Complete the form");
    }
    else if(password_1 !== password_2){
        window.alert("passwords need to be identical");
    }
    else
        $("form").submit();
}