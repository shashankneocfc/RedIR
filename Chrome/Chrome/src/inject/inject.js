

var readyStateCheckInterval = setInterval(function() {
	//console.log("test");
	if (document.readyState === "complete") {
		clearInterval(readyStateCheckInterval);
		// ----------------------------------------------------------
		// This part of the script triggers when page is done loading
		
		// ----------------------------------------------------------
		var href=location.href;
		var html=$("html").html();
	//	alert(html);
	//	console.log(html);
	//	alert(href);
		 chrome.extension.sendRequest({baseurl:href,htmlcontent:html});
		console.log("Hello. This message was sent from scripts/inject.js="+href);
		
		$("a").click(function() {
		var clicked_href=location.href;
		var clicked_html=$("html").html();
        chrome.extension.sendRequest({previousurl:clicked_href,previoushtml:clicked_html});
    });
	}
	}, 10);