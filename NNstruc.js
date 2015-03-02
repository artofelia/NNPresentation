
var b = document.getElementById("b");
var c = document.getElementById("c");
var ctx = c.getContext('2d');
//var ic = document.getElementById("ic"); //for drawing input
//var ictx = ic.getContext('2d');

var numPts = (Object.keys(raw).length-1)/2;
var row_sz = raw['rowsz']; //pixels in row of image
var inn = raw['0x'].length; //number of pixels in image
var outn = 1; //number of outputs
var layer_size = [3, 3]; //number of nuerons in each hidden layer
var L = layer_size.length; //number of hidden layers

//console.log('inn',inn);

var xv = []; //value of neurons
var init_xv = function(){
	xv.push(Matrix.Random(inn+1,1)); //constant in first row
	_.each(_.range(L), function(i){
		xv.push(Matrix.Random(layer_size[i],1));
		});
	xv.push(Matrix.Random(outn,1));
	
}

var w = [];
//matrix of weights between each layers of neurons
var init_w = function(){
	w.push([]);//index should start at 1
	var inp = Matrix.Random(inn+1, layer_size[0]-1);
	w.push(inp);
	var hidden = _.each(_.range(L-1), function(i) {
		w.push(Matrix.Random(layer_size[i], layer_size[i+1]-1));
	});
	var op = Matrix.Random(layer_size[L-1], outn);
	w.push(op);
}


var init_all = function(){
	console.log('initializing arrays');
	init_xv();
	//console.log('xv',xv);
	init_w();
	//console.log('w',w);
}

var set_inputs = function(numstr){
	xv[0].elements[0][0] = 1;
	_.each(_.range(numstr.length), function(i) {
		xv[0].elements[i+1][0] = parseInt(numstr[i]);
	});
	//console.log('setin', xv[0].transpose().inspect());
}

var forward = function(ind){
	var numstr = raw[ind + 'x'];
	
	set_inputs(numstr);
	_.each(_.range(L+2), function(l){ //each layer
		if (l==0) return;
		//console.log(l, w[l].transpose().dimensions(), xv[l-1].dimensions());
		var s = w[l].transpose().multiply(xv[l-1]);
		var theta = s.map( function(x, i, j) {
			return Math.tanh(x);
		});
		//console.log(s.inspect());
		if(l==L+1){
			xv[l] = theta.dup();
		}else {
			xv[l] = $M([[1].concat(theta.transpose().elements[0])]).transpose(); //add constant neuron
		}
		
	});
	return xv[xv.length-1];
}
//console.log('op',op.transpose().inspect());

var format_output = function(yn, type) {
	var res = Matrix.Zero(outn, 1);
	if(type=='digit'){
		res.elements[yn][0] = 1;
	}else if(type=='xor'){
		res.elements[0][0] = yn;
	}
	return res;
}

var del = [];

//del[a][b] - a=layer;b=neuron in layer, (a starts at 1), del[a] is a matrix as well 
var init_del = function(){
	del.push([]);
	_.each(_.range(L), function(i){
		del.push(Matrix.Random(layer_size[i]-1,1));
		});
	del.push(Matrix.Random(outn,1));
}
init_del();

var lrate = 0.1;//learning rate
var backprop = function(yn){
	var y = format_output(yn, 'xor'); //given in form of network outputs
	//console.log('y',y.transpose().inspect());
	//set deltas
	
	//output deltas
	//del[L+1].elements[j] = 2 * (xv[L+1].e(j+1,0) - y.e(j+1,0)) * (1 - Math.pow(xv[L+1].e(j+1,0), 2));
	var lastsub = xv[L+1].subtract(y);
	var lastcoeff = xv[L+1].map( function(x) {return x*x;} );
	del[L+1] = lastsub.map( function(x,i,j) {
		return 2 * x * (1-lastcoeff.e(i, j));
	});
	//hidden deltas
	//console.log('delL+1', del[L+1].transpose().inspect());
	_.each(_.range(L), function(sl){
		var l = L-sl; //going backwards
		var s = w[l+1].multiply(del[l+1]); //summation w[l][i][j]*del[l][j]
		var coeff = xv[l].map( function(x) {return x*x;} ); //for the derivative in chain rule
		var ndel = s.map( function(x, i, j){
			return (1-coeff.e(i,j)) * s.e(i,j); //calculate del[l-1][i]
		});
	});
	
	//update w
	_.each(_.range(L+1), function(ind) {
		var windch = ind+1;
		var ch = w[windch].map(function(x, i, j) {
			return lrate * xv[ind].e(i,0) * del[windch].e(j,0);
		});
		//console.log('wdim', ch.dimensions());
		w[windch] = w[windch].subtract(ch.dup());
	});
}

var xyToScreen = function(pt){
	return [pt[0] + c.width/2.0, -pt[1] + c.height/2.0];
}

var drawInOut = function(ind){
	ctx.font="20px Georgia";
	ctx.fillStyle="black";
	ctx.fillText("In:   " + raw[ind + 'x'],30,50);
	ctx.fillText("Out: " + raw[ind + 'y'],30,70);
}

var drawNetwork = function() {
	ctx.font="15px Georgia";
	var r = 20;
	var iposx = 100;
	var iposy = 100;
	var chposx = (c.width-2*iposx)/(L+2);
	
	var cposx = iposx;
	var cposy = iposy+100;
	var chposy = (c.height-2*iposy)/(inn+1);
	
	_.each(_.range(inn+1), function(i) {
		ctx.beginPath();
		ctx.strokeStyle="blue";
		ctx.arc(cposx,cposy,r,0,2*Math.PI);
		ctx.stroke();
		ctx.closePath();
		ctx.fillStyle="black";
		var xvval = Math.round(xv[0].e(i+1,1)*100)/100
		ctx.fillText("" + xvval,cposx-15,cposy+3);
		
		cposy += chposy;
	});
	
	_.each(_.range(L), function(l) {
		
		cposy = iposy+100;
		cposx += chposx;
		
		_.each(_.range(layer_size[l]), function(i) {
			ctx.beginPath();
			ctx.strokeStyle="green";
			ctx.arc(cposx,cposy,r,0,2*Math.PI);
			ctx.stroke();
			ctx.closePath();
			ctx.fillStyle="black";
			//console.log(l+1, i+1, 1, xv[l+1].e(i+1,1));
			var xvval = Math.round(xv[l+1].e(i+1,1)*100)/100
			ctx.fillText("" + xvval,cposx-15,cposy+3);
			
			cposy += chposy;
		});
	});
	cposx += chposx;
	cposy =((c.height-2*iposy)/(outn));
	_.each(_.range(outn), function(i) {
		ctx.beginPath();
		ctx.strokeStyle="blue";
		ctx.arc(cposx,cposy,r,0,2*Math.PI);
		ctx.stroke();
		ctx.closePath();
		ctx.fillStyle="black";
		var xvval = Math.round(xv[L+1].e(i+1,1)*100)/100
		ctx.fillText("" + xvval,cposx-15,cposy+3);
		
		cposy += chposy;
	});
	
	
}

var drawNumber = function(ind){
	var numstr = raw[ind + 'x'];
	//console.log(numstr);
	var imgData = ictx.createImageData(row_sz,row_sz); // only do this once per page
	for (var i=0;i<imgData.data.length;i+=4) {
		if (numstr[i/4]==1){
			imgData.data[i+0]=255;
			imgData.data[i+1]=255;
			imgData.data[i+2]=255;
			imgData.data[i+3]=255;
		}else if (numstr[i/4]==0){
			imgData.data[i+0]=0;
			imgData.data[i+1]=0;
			imgData.data[i+2]=0;
			imgData.data[i+3]=255;
		}
	}
	ctx.putImageData(imgData,10,10);
}

var auto = false;
b.onclick = function() { auto = !auto; };
init_all(); //initialize weight, and xv arrays
var tind = 1;

var update = function() {
	ctx.fillStyle="#ffffff";
	ctx.fillRect(0,0,c.width,c.height);
	drawInOut(tind);
	drawNetwork();
	if(auto) {
		forward(tind);
	}
	//drawNumber(tind);
	window.requestAnimationFrame(update);
}

var clicked = function(e) {
	var x = e.offsetX-300;
	var y = -e.offsetY+300;
	tind = Math.floor(numPts*Math.random());
	console.log('tind', tind);
	forward(tind);
	//backprop(raw[tind+'y']);
	
};

c.addEventListener("click",clicked);
window.requestAnimationFrame(update);
