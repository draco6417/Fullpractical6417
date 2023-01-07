1. Create an HTML form that contain the Student Registration details and write a
JavaScript to validate Student first and last name as it should not contain other than
alphabets and age should be between 18 to 50.
<html>
    <body>
    <h1 style="text-align: center;">REGISTRATION FORM</h1>
    <form name="RegForm" onsubmit="return GEEKFORGEEKS()" method="post">
 
        <p>Name: <input type="text" size="65" name="Name" /></p>
 
        <br />
 
        <p>Address: <input type="text" size="65" name="Address" />
        </p>
 
        <br />
 
        <p>E-mail Address: <input type="text" size="65" name="EMail" /></p>
 
        <br />
 
        <p>Password: <input type="text" size="65" name="Password" /></p>
 
        <br />
 
        <p>Telephone: <input type="text" size="65" name="Telephone" /></p>
 
        <br />
 
 
        <p>
            SELECT YOUR COURSE
            <select type="text" value="" name="Subject">
                <option>BTECH</option>
                <option>BBA</option>
                <option>BCA</option>
                <option>B.COM</option>
                <option>GEEKFORGEEKS</option>
            </select>
        </p>
 
        <br />
        <br />
 
        <p>Comments: <textarea cols="55" name="Comment"> </textarea></p>
 
 
        <p>
            <input type="submit" value="send" name="Submit" />
            <input type="reset" value="Reset" name="Reset" />
        </p>
 
    </form>
    <script>
        function GEEKFORGEEKS() {
            var name =
                document.forms.RegForm.Name.value;
            var email =
                document.forms.RegForm.EMail.value;
            var phone =
                document.forms.RegForm.Telephone.value;
            var what =
                document.forms.RegForm.Subject.value;
            var password =
                document.forms.RegForm.Password.value;
            var address =
                document.forms.RegForm.Address.value;
            var regEmail = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/g; //Javascript reGex for Email Validation.
            var regPhone = /^\d{10}$/;									 // Javascript reGex for Phone Number validation.
            var regName = /\d+$/g;								 // Javascript reGex for Name validation
 
            if (name == "" || regName.test(name)) {
                window.alert("Please enter your name properly.");
                name.focus();
                return false;
            }
 
            if (address == "") {
                window.alert("Please enter your address.");
                address.focus();
                return false;
            }
 
            if (email == "" || !regEmail.test(email)) {
                window.alert("Please enter a valid e-mail address.");
                email.focus();
                return false;
            }
 
            if (password == "") {
                alert("Please enter your password");
                password.focus();
                return false;
            }
 
            if (password.length < 6) {
                alert("Password should be atleast 6 character long");
                password.focus();
                return false;
 
            }
            if (phone == "" || !regPhone.test(phone)) {
                alert("Please enter valid phone number.");
                phone.focus();
                return false;
            }
 
            if (what.selectedIndex == -1) {
                alert("Please enter your course.");
                what.focus();
                return false;
            }
 
            return true;
        }
    </script>
 
</body>
</html>

2. Create an HTML form that contain the Employee Registration details and write a
JavaScript to validate DOB, Joining Date, and Salary.
<html>
 
<head>
 
    <title>Registration Form</title>
 
    <script type="text/javascript">
        function validate_form() {
            if (document.emp.emp_name.value == "") {
                alert("Please fill in the 'Your Employee Name' box.");
                return false;
            }
            if (document.emp.num.value == "") {
                alert("Enter Employee Number");
                return false;
            }
            alert("sucessfully Submitted");
 
 
 
            if (!(/^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/).test(document.emp.email_id.value)) {
                alert("You have entered an invalid email address!")
                return (false)
            }
        }
 
        function isNumberKey(evt) {
            var charCode = (evt.which) ? evt.which : event.keyCode;
            if (charCode != 46 && charCode > 31 &&
                (charCode < 48 || charCode > 57)) {
                alert("Enter Number");
                return false;
            }
            return true;
        }
 
 
        //-->
    </script>
 
</head>
 
<body bgcolor="#FFFFFF">
    <form name="emp" action="" onsubmit="return validate_form();" method="post">
        <table align="center" width=40% width="100%" cellspacing="2" cellpadding="2" border="5">
            <tr>
                <td colspan="2" align="center"><b>Employee Registration Form.</b></td>
 
            </tr>
            <tr>
                <td align="left" valign="top" width="41%">Employee Name<span style="color:red">*</span></td>
 
                <td width="57%"><input type="text" value="" name="emp_name" size="24"></td>
            </tr>
            <tr>
                <td align="left" valign="top" width="41%">Employee Number<span style="color:red">*</span></td>
                <td width="57%">
                    <input type="text" value="" name="num" onkeypress="return isNumberKey(event)" size="24"></td>
            </tr>
            <tr>
                <td align="left" valign="top" width="41%">Address</td>
 
                <td width="57%"><textarea rows="4" maxlen="200" name="S2" cols="20"></textarea></td>
            </tr <tr>
 
            <td align="left" valign="top" width="41%">Contact Number</td>
            <td width="57%">
                <input type="text" value="" onkeypress="return isNumberKey(event)" name="txtFName1" size="24"></td>
            </tr>
            <tr>
                <td align="left" valign="top" width="41%">Job Location</td>
                <td width="57%"><select name="mydropdown">
<option value="Default">Default</option>
<option value="Chennai">Chennai</option>
<option value="Bangalore">Bangalore</option>
<option value="Chennai">Pune</option>
<option value="Bangalore">Mysore</option>
<option value="Chennai">Chandigarh</option>
 
</select></td>
 
 
            </tr>
 
            <tr>
                <td align="left" valign="top" width="41%">Designation</td>
                <td width="57%">
                    <select name="mydropdown">
<option value="Default">Default</option>
<option value="Systems Engineer">Systems Engineer</option>
<option value="Senior Systems Engineer">Senior Systems Engineer</option>
<option value="Technology Analyst">Technology Analyst</option>
<option value="Technology Lead">Technology Lead</option>
<option value="Project Manager">Project Manager</option>
 
 
</select></td>
 
 
            </tr>
            <tr>
                <td align="left" valign="top" width="41%">Email<span style="color:red">*</span></td>
                <td width="57%">
                    <input type="text" value="" name="email_id" size="24"></td>
            </tr>
 
            <tr>
                <td colspan="2">
                    <p align="center">
                        <input type="submit" value="  Submit" name="B4">               
                        <input type="reset" value="  Reset All   " name="B5"></td>
            </tr>
 
        </table>
    </form>
</body>
 
</html>

3. Create an HTML form for Login and write a JavaScript to validate email ID using
Regular Expression.
<html>
<head>
    <title>Javascript Login Form Validation</title>
    <!-- Include JS File Here -->
    <script src="login.js"></script>
</head>


<body>
    <div class="container">
        <div class="main">
            <h2>Javascript Login Form Validation</h2>
            <form id="form_id" method="post" name="myform">
                <label>User Name :</label>
                <input type="text" name="username" id="username" />
                <label>Password :</label>
                <input type="password" name="password" id="password" />
                <input type="button" value="Login" id="submit" onclick="validate()" />
            </form>
            <span><b class="note">Note : </b>For this demo use following username and password. <br /><b
                    class="valid">User Name : Formget<br />Password : formget#123</b></span>
        </div>
    </div>
</body>
</html>
Login.js
var attempt = 3; // Variable to count number of attempts.
// Below function Executes on click of login button.
function validate(){
var username = document.getElementById("username").value;
var password = document.getElementById("password").value;
const res = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
var us_name = res.test(String(username).toLowerCase());
if(us_name)
{
    if ( username == "formget@gmail.com" && password == "formget#123"){
        alert ("Login successfully");
        return false;
    }
    else{
        attempt --;// Decrementing by one.
        alert("You have left "+attempt+" attempt;");
        // Disabling fields after 3 attempts.
        if( attempt == 0){
            document.getElementById("username").disabled = true;
            document.getElementById("password").disabled = true;
            document.getElementById("submit").disabled = true;
            return false;
        }
    }
}
}

4. Create a Node.js file that will convert the output "Hello World!" into upper-case letters:
var http=require('http');
var uc=require('upper-case');
http.createServer(function(req,res){
    res.writeHead(200,{'Content-Type':'text/html'});
    res.write(uc.upperCase("hello world good morning"));
    res.end();
}).listen(8080);

5. Using nodejs create a web page to read two file names from user and append contents
of first file into second file
var http = require('http');
var fs = require('fs');
var formidable = require('formidable');


http.createServer(function (req, res) {
    if(req.url=='/'){
        res.writeHead(200, {'Content-Type': 'text/html'});
        res.write('<form action="fapp" method="post" enctype="multipart/form-data">');
        res.write('<h1> SELECT TWO FILNES</h1>');
        res.write('<input type="file" name="rf"><br>');
        res.write('<input type="file" name="wf"><br>');
        res.write('<input type="submit">');
        res.end();


    }
    else if(req.url=='/fapp'){
        var form=new formidable.IncomingForm();
        form.parse(req,function(err,fields,files) 
        {
            if(!err){
                var w=fs.createWriteStream(files.wf.name,{flags:'a'});
                var r=fs.createReadStream(files.rf.name);
                w.on('close',function(){
                    console.log("Writing Done");
                });
                r.pipe(w);
                res.write(files.rf.name);
                res.end("Appended successfully");
            }
            else{res.write("Error in Writing");}
        });
    }
    else{
        res.end("Page not found");
    }
  
}).listen(8080);

6. Create a Node.js file that opens the requested file and returns the content to the client.
If anything goes wrong, throw a 404 error
var http = require('http');
var fs = require('fs');


http.createServer(function(req,res){
    fs.open('input.txt','r+', function(err,fd){
        if(err){
            console.error(err);
            return res.end("404 Not found");
        } 
        else{
            console.error("File opened successfully");
            fs.readFile('sample.txt',function(err, data) { 
                if(!err){
                    console.log('success');
                    res.end(data);
                    fs.close(fd);
                }});
            }
        });
        }).listen(5000);

7. Create a Node.js file that writes an HTML form, with an upload field
var http=require('http');
var fs=require('fs');
var formidable=require('formidable');


http.createServer(function(req,res){
    if(req.url=='/'){
        res.writeHead(200,{'content-type':'text/html'});
        res.write('<form action="fapp" method="post" enctype="multipart/form-dat">');
        res.write('<h1>Registration</h1>');
        res.write('Applicant Nmae: <input type="text" name="t1"><br>');
        res.write('Phone: <input type="text" name="t2"><br>');
        res.write('Address: <input type="text" name="t3"><br>');
        res.write('Resume Upload: <input type="file" name="filetoupload"><br>');
        res.write('<input type="submit" name="upload"><br>');
    }
    else if(req.url=='/fapp'){
        var form= new formidable.IncomingForm();
        form.parse(req,function(err,fields,files){
            res.write('<h1>Name:' +fields.t1 + '</h1>');
            res.write('<h1>Phone:' +fields.t2 + '</h1>');
            res.write('<h1>Address:' +fields.t3 + '</h1>');


            var oldpath =files.filetoupload.path;
            var newpath ='./' + files.filetoupload.newFilename;


            fs.rename(oldpath, newpath, function(err){
                if(err) throw err;
                else{
                    res.write('Resume Upload and moved successfully');
                    res.end();
                }
            });
        });
    }
    else{ res.end("Page not found");}
}).listen(80);

8. Create a Node.js file that demonstrate create database and table in MySQL
var mysql=require('mysql');


var con=mysql.createConnection({
    host:"localhost",
    user:"root",
    password:""
});


con.connect(function(err){
    if(err) throw err;
    console.log("connected");
    con.query("CREATE DATABASE inventory", function(err,result){
        if(err) throw err;
        else{
            console.log("Database Created");
            con.query("use inventory");
            var sql = "CREATE TABLE customer(cid INT,cname VARCHAR(25), phone INT(10), city VARCHAR(25))";
            con.query(sql, function(err,result){
                if(err) throw err;
                console.log("Table Created");
            });
        }
    });
});
9. Create a node.js file that Select all records from the "customers" table, and display the
result object on console
var sql=require('mysql');


var con=mysql.createConnection({
    host:"localhost",
    user:"root",
    password:""
});


con.connect(function(err){
    if(err) throw err;
    con.query("SELECT * FROM customers", function(err, result,fields){
        if(err) throw err;
        console.log(result);
    });
});

10. Create a node.js file that Insert Multiple Records in "student" table, and display the
result object on console
var mysql=require('mysql');


var con=mysql.createConnection({
    host:"localhost",
    user:"root",
    password:"",
    port: 3000,
    database: "studentdb"
});


con.connect(function(err){
    if(err) throw err;
    var records=[
        ['Arun',25,9887765],
        ['Jack',16,63844848],
        ['Priya',17,84747383],
        ['Amy',15,99828282]
    ];
con.query("INSERT INTO students(name, rollno, marks) VALUES ?",[records], function(err,result,fields){
    if(err) throw err;


    console.log(result);
    console.log("Number of rows affected :" + result.affectedRows);
    console.log("Number of records affected with warning : " + result.warningCount);
    console.log("Message from Mysql server : " + result.message);


});


});

11. Create a node.js file that Select all records from the "customers" table, and delete the
specified record.
var mysql=require('mysql');


var con=mysql.createConnection({
    host:"localhost",
    user:"root",
    password:"",
    database: "inventory"
});


con.connect(function(err){
    if(err) throw err;
    con.query("SELECT * FROM customers",function(err,result,fields){
        if(err) throw err;
        console.log(result);
    });
    var sql="DELETE FROM customers WHERE city='Mumbai'";
    con.query(sql, function(err, result){
        if(err) throw err;
        console.log("Number of records deleted : " + result.affectedRows);
    })
    
});



12. Create a Simple Web Server using node js
var http=require('http');
var server=http.createServer(function(req,res){
    res.writeHead(200, {'content-Type': 'text/html'});
    res.write("Hello Node JS");
    res.end("THE END");
});
server.listen(5000);

13. Using node js create a User Login System
var http=require('http');
var fs=require('fs');
var formidable=require('formidable');
var mysql=require('mysql');


var con=mysql.createConnection({
    host:"localhost",
    user:"root",
    password:"",
    database: "studentdb"
});


http.createServer(function(req,res){
    if(req.url=='/'){
        res.writeHead(200, {'content-Type':'text/html'});
        res.write('<form action="fapp" method="post" enctype="multipart/form-data">');
        res.write('<h1>Registration Form</h1><br>');
        res.write('User Name : <input type="text" name="t1"><br>');
        res.write('Password : <input type="text" name="t2"><br>');
        res.write('<input type="button" value="LOGIN"><br>');
        res.end();
        
    }
    else if(req.url=='/fapp'){
        var form=new formidable.IncomingForm();
        form.parse(req, function(err,fields,files){
            var un=fields.t1;
            var pass=fields.t2;
            res.write('<h1><center>Hello : ' + un + '</center></h1><br>');
            con.connect(function(err){
                if(!err){
                    con.query("SELECT * FROM login where uname = ? and password = ?" , [un, pass], function(err,result,fields){
                        if(result.length>0){
                            res.end('<h1>LOGIN SUCCESS</h1>');
                        }
                        else{
                            res.end('<h1>User Name and Password not matching</h1>');
                        }
                    });
                }
            });
        });
    }
    else{
        res.end("Page Not Found");
    }
}).listen(8020);

14. Using node js create a eLearning System
Html
<html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <style>
        body{
            font-family: Arial;
            color: white;
        }
        .split1{
            height: 100%;
            width: 35%;
            position: fixed;
            z-index: 1;
            top: 0;
            overflow: scroll;
            padding-top: 20px;
        }
        .split1{
            height: 100%;
            width: 65%;
            position: fixed;
            z-index: 1;
            top: 0;
            overflow-x: hidden;
            padding-top: 20px;
        }
        .left{
            left: 0;
            background-color: #111;
        }
        .right{
            right: 0;
            background-color: red;
        }
        .centered{
            position:relative;
            transform: translate(-50%, -50%);
            top: 50%;
            left: 20%;
            text-align: center;
        }
        .link{
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: 1;


        }
    </style>
    <body>
        <div class="split1 left">
            <div class="centered">
                <h1>
                    <a href="/html_tutorial" target="a">HTML</a><br>
                    <a href="/nodejs_tutorial" target="a">Node JS</a><br>
                    <a href="/javascript_tutorial" target="a">JavaScript</a><br>
                    <a href="/css_tutorial" target="a">CSS</a><br>
                </h1>
            </div>
            </div>
            <div class="split1 right">
                <iframe name="a" height="100%" width="100%"></iframe>
            </div>
    </body>
</html>
Js
var fs=require('fs');
var http =require('http');


var con=http.createServer(function(req,res){
    if(req.url=='/'){
        fs.readFile('slip14.html',function(err,data){
            res.writeHead(200, {'content-Type': 'text/html'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url=='/html_tutorial'){
        fs.readFile('html_tutorial.pdf',function(err,data){
            res.writeHead(200, {'content-Type': 'application/pdf'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url=='/nodejs_tutorial'){
        fs.readFile('nodejs_tutorial.pdf',function(err,data){
            res.writeHead(200, {'content-Type': 'application/pdf'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url=='/javascript_tutorial'){
        fs.readFile('javascript_tutorial.pdf',function(err,data){
            res.writeHead(200, {'content-Type': 'application/pdf'});
            res.write(data);
            res.end();
        });
    }
     else if(req.url=='/css_tutorial'){
        fs.readFile('css_tutorial.pdf',function(err,data){
            res.writeHead(200, {'content-Type': 'application/pdf'});
            res.write(data);
            res.end();
        });
    }
    else{
        res.end("The end");
    }
}).listen(3000);
console.log("SUCCESS");

15. Using node js create a Recipe Book
var fs=require('fs');
var http=require('http');
var con=http.createServer(function(req,res){
    if(req.url=='/'){
        fs.readFile('slip15.html', function(err,data){
            res.writeHead(200,{'Content-Type':'text/html'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url.match("./jpg$")){
         var filestream=fs.createReadStream("menu.jpg");
         res.writeHead(200,{'Content-Type':'image/jpg'});
         filestream.pipe(res);
    }
    else if(req.url.match("./png$")){
         var filestream=fs.createReadStream("samosa1.png");
         res.writeHead(200,{'Content-Type':'image/png'});
         filestream.pipe(res);
    }
    else if(req.url=="/contact"){
        fs.readFile('contact.html', function(err,data){
            res.writeHead(200,{'Content-Type':'text/html'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url=="/about"){
        fs.readFile('about.html', function(err,data){
            res.writeHead(200,{'Content-Type':'text/html'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url=="/snacks"){
        fs.readFile('Snacks.pdf', function(err,data){
            res.writeHead(200,{'Content-Type':'application/pdf'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url=="/cake"){
        fs.readFile('cake.pdf', function(err,data){
            res.writeHead(200,{'Content-Type':'application/pdf'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url=="/rice"){
        fs.readFile('Rice.pdf', function(err,data){
            res.writeHead(200,{'Content-Type':'application/pdf'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url=="/chicken"){
        fs.readFile('Chicken.pdf', function(err,data){
            res.writeHead(200,{'Content-Type':'application/pdf'});
            res.write(data);
            res.end();
        });
    }
    else{
        res.end("THE END");
    }
    
}).listen(3007);
console.log("SUCCESS");


<html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        <style>
            body {
            font-family: "Lato", sans-serif;
            background-color: pink;
}


/* Fixed sidenav, full height */
.sidenav {
    margin: 0;
    height: 100%;
    width: 200px;
    position: fixed; 
    background-color: #f1f1f1;
    overflow-x: auto;
    padding: 0;
}
/* Style the sidenav links and the dropdown button */
.sidenav a, .dropdown-btn {
  padding: 6px 8px 6px 16px;
  text-decoration: none;
  font-size: 20px;
  color:none;
  display: block;
  border: none;
  background: none;
  width: 100%;
  text-align: left;
  cursor: pointer;
  outline: none;
}


/* On mouse-over */
.sidenav a:hover, .dropdown-btn:hover {
  color: blue;
}


/* Main content */
.main {
  margin-left: 200px; /* Same as the width of the sidenav */
  font-size: 20px; /* Increased text to enable scrolling */
  padding: 0px 10px;
}


/* Add an active class to the active dropdown button */
.active {
  background-color: green;
  color: rgb(255, 0, 149);
}


/* Dropdown container (hidden by default). Optional: add a lighter background color and some left padding to change the design of the dropdown content */
.dropdown-container {
  display: none;
  background-color: orange;
  padding-left: 8px;
}


/* Optional: Style the caret down icon */
.fa-caret-down {
  float: right;
  padding-right: 8px;
}
/* Some media queries for responsiveness */
@media screen and (max-height: 450px) {
  .sidenav {padding-top: 15px;}
  .sidenav a {font-size: 18px;}
}


div.content {
    background-color: #999;
    margin-left: 200px;
    padding: 1px 16px;
    height: 1000px;
    
}
        </style>
    </head>
    <body>
        <center><h1>Recipes in my Way</h1></center>
        <div class="sidenav">
            <a class="active" href="samosa1.jpg" target="a">Home</a>
            <button class="dropdown-btn">Delicious Menus
                <i class="fa fa-caret-down"></i>
            </button>
            <div class="dropdown-container">
                <a href="/rice" target="a">Rice</a>
                <a href="/snacks" target="a">Snacks</a>
                <a href="/cake" target="a">Cake</a>
                <a href="/chicken" target="a">Chicken</a>
            </div>
            <a href="/contact" target="a">Contact</a>
            <a href="/about" target="a">About</a>
        </div>
        <div class="content">
            <iframe src="menu.jpg" name="a" height="100%" width="100%"></iframe>
        </div>
        <script>
            //* Loop through all dropdown buttons to toggle between hiding and showing its dropdown content - This allows the user to have multiple dropdowns without any conflict */
                var dropdown = document.getElementsByClassName("dropdown-btn");
                var i;


                for (i = 0; i < dropdown.length; i++) {
                    dropdown[i].addEventListener("click", function () {
                        this.classList.toggle("active");
                        var dropdownContent = this.nextElementSibling;
                        if (dropdownContent.style.display === "block") {
                            dropdownContent.style.display = "none";
                        } else {
                            dropdownContent.style.display = "block";
                        }
                    });
                }
        </script>
    </body>
</html>
<html>
    <head>
        <style>
            h1{
                font-family: serif;
                font-size: 5em;
                color: red;
                text-align: center;
                animation: animate 1.5s linear infinite;
            }
            @keyframes animate {
                0%{
                    opacity: 0;
                }
                50%{
                    opacity: 0.7;
                }
                100%{
                    opacity: 0;
                }
            }
        </style>
    </head>
    <body>
        <center>
            <h1>Recipes in my way</h1>
            <h3>Mouth Watering Food</h3>
            <marquee bgcolor="pink" behavior="alternate">
                <h2>For Home Delivery Contact: 988776663</h2>
            </marquee>
             <h4>Address : Pune</h4>
        </center>
    </body>
</html>
<html>
    <head>
        <style>
            h1{
                font-family: serif;
                font-size: 5em;
                color: red;
                text-align: center;
                animation: animate 1.5s linear infinite;
            }
            @keyframes animate {
                0%{
                    opacity: 0;
                }
                50%{
                    opacity: 0.7;
                }
                100%{
                    opacity: 0;
                }
            }
        </style>
    </head>
    <body>
        <center>
            <h1>Recipes in my way</h1>
            <h3>Mouth Watering Food</h3>
            <marquee bgcolor="pink" behavior="alternate">
                <h2>For Home Delivery Contact: 988776663</h2>
            </marquee>
             <h4>Address : Pune</h4>
        </center>
    </body>
</html>

16. write node js script to interact with the filesystem, and serve a web page from a
File
<html>
    <head>
        <style>
            .hero-image{
                background-color: gray;
                height:500px;
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
                position: relative;
            }
            .hero-text{
                text-align: center;
                position: absolute;
                top: 20%;
                left: 20%;
                font-size: x-large;
                transform: translate(-50%, -50%);
                color: honeydew;
            }
        </style>
    </head>
     <body>
        <div class="hero-image">
            <img src="./nodejs.jpg">
            <div clas="hero-text">
                <script>
                    var nm=prompt("Enter your Nmae");
                    document.write("NodeJs is easy to learn "+ nm +"...");
                </script>
            </div>
        </div>
     </body>
</html>
var http=require('http');
var fs=require('fs');
var con=http.createServer(function(req,res){
    if(req.url=='/'){
        fs.readFile('slip16.html',function(err,data){
            res.writeHead(200,{'content-Type':'text/html'});
            res.write(data);
            res.end();
        });
    }
    else if(req.url.match("\.jpg")){
        var filestream=fs.createReadStream("nodejs.jpg");
        res.writeHead(200,{'content-Type':'text/html'});
        filestream.pipe(res);
    }
    else{
        res.end();
    }
}).listen(4567);

17. Write node js script to build Your Own Node.js Module. Use require (‘http’)
module is a built-in Node module that invokes the functionality of the HTTP
library to create a local server. Also use the export statement to make functions
in your module available externally. Create a new text file to contain the
functions in your module called, “modules.js” and add this function to return
today’s date and time.
Modules.js
function datetime(){
    let dt=new Date();
    let date=("0" + dt.getDate()).slice(-2);
    let month=("0" +(dt.getMonth() + 1)).slice(-2);
    let year=dt.getFullYear();
    let hours=dt.getHours();
    let Minutes=dt.getMinutes();
    let seconds=dt.getSeconds();
    
    var output=year + "-" + month + "-" + date + " " + hours + ":" + Minutes + ":" + seconds;
    return output;
}
module.exports={datetime};
Slip17.js
var http = require('http');


var dt=require('./modules.js');
var server=http.createServer(function(req,res){
    res.writeHead(200,{'Content-Type': 'text/html'});
    const result=dt.datetime();
    res.write('Current Date and Time');
    res.write(result);
    res.end();
});
server.listen(1234);

18. Create a js file named main.js for event-driven application. There should be a
main loop that listens for events, and then triggers a callback function when one
of those events is detected.
var events=require('events');


var eventEmitter= new events.EventEmitter();


var connectHandler=function connected(s){
    console.log('Its' ,s);
}


eventEmitter.on('data_received',function(name){
    console.log(name,"Understood Event-Driven");
});
eventEmitter.emit('Data-received',"PETER");


eventEmitter.on('connetion', connectHandler);
eventEmitter.emit('connection',"SIMPLE SOLUTION");


console.log("Program Ended");



 19. Write node js application that transfer a file as an attachment on web and enables
browser to prompt the user to download file using express js.
<html>
    <body bgcolor="orange">
        <form action="/file-data" method="post">
            <center>
                <table border="1">
                    <tr>
                        <td>Select a file</td>
                        <td><input type="file" name="id"></td>
                    </tr>
                    <tr align="center">
                        <td colspan="2"><input type="submit" value="Download"></td>
                    </tr>
                </table>
            </center>
        </form>
    </body>
</html>
var express=require('express');
const fs=require('fs');
var app=express();
var PORT=1234;


var bodyParser=require("body-parser");
app.use(bodyParser.urlencoded({extended:false}));
app.get('/',function(req,res){
    const files=fs.createReadStream('slip13.html');
    res.writeHead(200, {'content-Type':'text/html'});
    files.pipe(res);
});


app.post('file-data',function(req,res){
    var name=req.body.id;
    console.log(name);
    res.download(name);
});


app.listen(PORT,function(err){
    if(err) console.log(err);
    console.log("SERVER running ", PORT);
});

20. Create your Django app in which after running the server, you should see on the
browser, the text “Hello! I am learning Django”, which you defined in the index view.
//urls.py
from django.contrib import admin
from django.urls import path
from django.conf.urls import include


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('hello.urls')),
]
//urls.py/Hello
from django.urls import path
from . import views
urlpatterns=[
    path('',views.index,name='index')
]

//views.py
from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
def index(request):
    return HttpResponse("Hello! I am learning Django")

Or
from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
TEMPLATES_DIRS=('os.path.join(BASE_DIR,"templates")')
def index(request):
    return render(request,'index.html')




21. Design a Django application that adds web pages with views and templates.
22. Write and run Django code to add data to your site using relational databases with
Django’s Object Relational Mapper.
23. Develop a basic poll application (app).It should consist of two parts:
a) A public site in which user can pick their favourite programming language and vote.
b) An admin site that lets you add, change and delete programming languages.
24. A public site in which user can pick their favourite programming language and vote.
25. An admin site that lets you add, change and delete programming languages.
26. Implement a simple Django application for portfolio management.
27. Create your own blog using Django
28. Build your own To-Do app in Django
29. Create a clone of the “Hacker News” website.
30. Develop Online School System using Django
31. Implement your E-commerce Website using Django
32. Implement Login System using Django