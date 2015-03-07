// if you checked "fancy-settings" in extensionizr.com, uncomment this lines

// var settings = new Store("settings", {
//     "sample_setting": "This is how you use Store.js to remember values"
// });


//example of using a message handler from the inject scripts

var db = window.openDatabase('mydb523', '1.0', 'my first database', 5 * 1024 * 1024);
var prev_html="";
var prev_url="";
chrome.extension.onRequest.addListener(
  function(request, sender) {
// alert(request.baseurl);
	//alert("sds");
	if(request.baseurl)
	{
		console.log("New webpage opened for url="+request.baseurl);
		db.transaction(function (tx) {
		//	console.log("DbBaseUrl is:"+request.baseurl);
			tx.executeSql('CREATE TABLE IF NOT EXISTS mar6 (ID INTEGER PRIMARY KEY ASC, url TEXT, html TEXT, previousurl TEXT , previoushtml TEXT)');
			tx.executeSql("INSERT INTO mar6 (url,html,previousurl,previoushtml) VALUES (?,?,?,?)",[request.baseurl,request.htmlcontent,prev_url,prev_html]);
	});
	
	}
    if(request.previousurl && request.previoushtml)
	{
		console.log("last url clicked stored:"+request.previousurl);
		prev_html=request.previoushtml;
		prev_url=request.previousurl;
	}
  
  });