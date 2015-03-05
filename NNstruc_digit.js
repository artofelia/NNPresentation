
var b = document.getElementById("b");
var c = document.getElementById("c");
var ctx = c.getContext('2d');
var ic = document.getElementById("ic"); //for drawing input
var ictx = ic.getContext('2d');

var numPts = (Object.keys(raw).length-1)/2;
var row_sz = raw['rowsz']; //pixels in row of image
var inn = raw['0x'].length; //number of pixels in image
var outn = 10; //number of outputs
var layer_size = [10*10, 10*10]; //number of nuerons in each hidden layer
var L = layer_size.length; //number of hidden layers
var my_guess = -1;
var iter = 0;
var num_right = 0;
var per_error = 0;

var xv = []; //value of neurons
var init_xv = function(){
	xv.push(Matrix.Zero(inn+1,1)); //constant in first row
	_.each(_.range(L), function(i){
		xv.push(Matrix.Zero(layer_size[i],1));
		});
	xv.push(Matrix.Zero(outn,1));
};

var w = [];
//matrix of weights between each layers of neurons
var init_w = function(){
	w.push([]);//index should start at 1
	var inp = Matrix.Random(inn+1, layer_size[0]-1).x(0.0001);
	w.push(inp);
	var hidden = _.each(_.range(L-1), function(i) {
		w.push(Matrix.Random(layer_size[i], layer_size[i+1]-1).x(0.0001));
	});
	var op = Matrix.Random(layer_size[L-1], outn);
	w.push(op);
}

var init_w_xor = function() {
	w.push([]);
	var M = $M([[-1.5, -1.5],
				[1, -1],
				[-1, 1]]);
	w.push(M);
	M = $M([[1.5],[1],[1]]);
	w.push(M);
}

var del = [];
//del[a][b] - a=layer;b=neuron in layer, (a starts at 1), del[a] is a matrix as well 
var init_del = function(){
	del.push([]);
	_.each(_.range(L), function(i){
		del.push(Matrix.Zero(layer_size[i]-1,1));
		});
	del.push(Matrix.Zero(outn,1));
}

var init_all = function(){
	console.log('initializing arrays');
	init_xv();
	//console.log('xv',xv);
	init_w();
	//console.log('w',w);
	init_del();
	setNodePos();
}

var set_inputs = function(numstr, type){
	xv[0].elements[0][0] = 1;
	if(type=='digit'){
		_.each(_.range(numstr.length), function(i) {
			xv[0].elements[i+1][0] = parseInt(numstr[i])*2-1;
		});
	}else if(type=='xor'){
		_.each(_.range(numstr.length), function(i) {
			xv[0].elements[i+1][0] = parseInt(numstr[i])*2-1;
		});
	}
}

var forward = function(ind){
	var numstr = raw[ind + 'x'];
	
	set_inputs(numstr, 'digit');
	_.each(_.range(L+2), function(l){ //each layer
		if (l==0) return;
		//console.log('dimcheck',l, w[l].transpose().dimensions(), xv[l-1].dimensions());
		var s = w[l].transpose().multiply(xv[l-1]);
		//console.log('s',s.inspect());
		//console.log('xvs',l-1,xv[l-1].transpose().inspect());
		var theta = s.map( function(x, i, j) {
			return Math.tanh(x); //has max performance limit
		});
		//console.log('sb', l, xv[l-1].inspect());
		//console.log('s', l, s.inspect());
		if(l==L+1){
			xv[l] = theta.dup();
		}else {
			xv[l] = $M([[1].concat(theta.transpose().elements[0])]).transpose(); //add constant neuron
		}
	});
	return xv[L+1];
}

var format_output = function(yn, type) {
	var res = Matrix.Zero(outn, 1);
	if(type=='digit'){
		res = res.map(function(){return -1;});
		res.elements[yn][0] = 1;
	}else if(type=='xor'){
		res.elements[0][0] = yn;
	}
	return res;
}

var lrate = 0.1;//learning rate
var start = new Date().getTime();
var end = -1;
var backprop = function(ind){
	var yn = raw[ind + 'y'];
	var y = format_output(yn, 'digit'); //given in form of network outputs
	//console.log('y',y.transpose().inspect());
	//set deltas
	
	//output deltas
	var lastsub = xv[L+1].subtract(y);
	console.log('error,lastsub', lastsub.transpose().inspect());
	var lastcoeff = xv[L+1].map( function(x) {return x*x;} );
	console.log('lastcoeff', lastcoeff.transpose().inspect());
	del[L+1] = lastsub.map( function(x,i,j) {
		return 2 * x * (1-lastcoeff.e(i, j));
	});
	console.log('delL+1', del[L+1].transpose().inspect());
	
	//hidden deltas
	_.each(_.range(L), function(sl){
		var l = L-sl; //going backwards
		//console.log('lind',l);
		var s = w[l+1].multiply(del[l+1]); //summation w[l][i][j]*del[l][j]
		var sem = s.transpose().elements[0];
		sem.splice(0,1);
		s = $M([sem]).transpose();
		var coeff = xv[l].map( function(x) {return x*x;} ); //for the derivative in chain rule
		//console.log('coeff', coeff.inspect());
		var ndel = s.map( function(x, i, j){
			var nval = (1-coeff.e(i+1,j)) * s.e(i,j);
			//console.log(coeff.e(i+1,j), s.e(i,j), i, j);
			return nval; //calculate del[l-1][i]
		});
		//console.log('s', s.inspect());
		//console.log('ndel', ndel.inspect());
		del[l] = ndel.dup();//remember first one is not used.
		//end = new Date().getMilliseconds();
		//console.log('hiddendeltime: ' + (end-start));
	});

	//update w
	_.each(_.range(L+1), function(ind) {
		//start = new Date().getMilliseconds();
		var windch = ind+1;
		//console.log('wupind', windch);
		var ch = xv[windch-1].multiply(del[windch].transpose()).x(lrate);
		w[windch] = w[windch].subtract(ch.dup());
		//end = new Date().getMilliseconds();
		//console.log('bfmap_hiddendeltime: ' + (end-start));
		//console.log('chdim', ch.e(1,1));
		//console.log('wdim', w[windch].dimensions());
	});
	//console.log('update_in', del[3].inspect(), w[3].e(1,1));
}

var xyToScreen = function(pt){
	return [pt[0] + c.width/2.0, -pt[1] + c.height/2.0];
}

var node_pos = [];
var setNodePos = function(){
	var iposx = 70;
	var iposy = 70;
	var chposx = (c.width-2*iposx)/(L+2);
	var cposx = iposx;
	var cposy = iposy+100;
	var chposy = (c.height-2*iposy)/(inn+1);
	
	node_pos[0] = [];
	_.each(_.range(inn+1), function(i) {
		node_pos[0].push([cposx, cposy]);
		cposy += chposy;
	});
	_.each(_.range(L), function(l) {
		
		cposy = iposy+100;
		cposx += chposx;
		node_pos[l+1] = [];
		_.each(_.range(layer_size[l]), function(i) {
			node_pos[l+1].push([cposx, cposy]);
			cposy += chposy;
		});
	});
	cposx += chposx;
	cposy =((c.height-2*iposy)/(outn));
	node_pos[L+1] = [];
	_.each(_.range(outn), function(i) {
		node_pos[L+1].push([cposx, cposy]);
		cposy += chposy;
	});
}

var drawInOut = function(ind){
	ctx.font="20px Georgia";
	ctx.fillStyle="black";
	var dy = 50;
	//ctx.fillText("In:   " + raw[ind + 'x'], 30, dy);
	ctx.fillText("Out: " + raw[ind + 'y'], 30, dy+20);
	ctx.fillText("Guess: " + my_guess, 30, dy+40);
	ctx.fillText("Iter: " + iter, 30, dy+60);
	ctx.fillText("Percent Error: " + per_error, 30, dy+80);
}

function scaleImageData(imageData, scale) {
    var scaled = ctx.createImageData(imageData.width * scale, imageData.height * scale);
    var subLine = ctx.createImageData(scale, 1).data
    for (var row = 0; row < imageData.height; row++) {
        for (var col = 0; col < imageData.width; col++) {
            var sourcePixel = imageData.data.subarray(
                (row * imageData.width + col) * 4,
                (row * imageData.width + col) * 4 + 4
            );
            for (var x = 0; x < scale; x++) subLine.set(sourcePixel, x*4)
            for (var y = 0; y < scale; y++) {
                var destRow = row * scale + y;
                var destCol = col * scale;
                scaled.data.set(subLine, (destRow * scaled.width + destCol) * 4)
            }
        }
    }

    return scaled;
}

var drawRep = function(l, cx, cy, sc){
	var sz = parseInt(Math.sqrt(layer_size[l-1]));
	var mx = 1.0;//xv[l].max();
	var imgData = ictx.createImageData(sz,sz); // only do this once per page
	for (var i=0;i<imgData.data.length;i+=4) {
		var cval = xv[l].e(i/4+1,1)*130.0+130;
		imgData.data[i+0]= cval;
		imgData.data[i+1]= cval;
		imgData.data[i+2]= cval;
		imgData.data[i+3]=255;
	}
	//console.log(xv[l].e(100+1,1));
	ctx.putImageData(scaleImageData(imgData,sc),cx,cy);
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
	ctx.putImageData(scaleImageData(imgData,3),50,150);
}

var auto = false;
b.onclick = function() {
	auto = !auto;
};
init_all(); //initialize weight, and xv arrays
var tind = 1;

var get_guess = function(outp) {
	console.log(outp);
	var maxi = -1;
	var mx = -1000;
	_.each(_.range(outp.length), function(i) {
		if(outp[i]>mx){
			mx = outp[i];
			maxi = i;
		}
	});
	return maxi;
}
var step = function(){
	tind = Math.floor(numPts*Math.random());
	forward(tind);
	
	var retf = forward(tind);
	my_guess = get_guess(retf.transpose().elements[0]);
	//start = new Date().getMilliseconds();
	backprop(tind);
	//end = new Date().getMilliseconds();
	//console.log('tbacktime', (end-start));
	var yn = raw[tind + 'y'];
	if(my_guess==yn){
		num_right++;
	}
	iter++;
	per_error = Math.round((1-num_right/parseFloat(iter))*100)/100.0;
}

var update = function() {
	ctx.fillStyle="#ffffff";
	ctx.fillRect(0,0,c.width,c.height);
	if(auto) {
		step();
	}
	drawInOut(tind);
	//drawNetwork();
	drawNumber(tind);
	drawRep(1, 150, 150, 10);
	drawRep(2, 300, 150, 10);
	window.requestAnimationFrame(update);
}

var clicked = function(e) {
	var x = e.offsetX-300;
	var y = -e.offsetY+300;
	step();
};

c.addEventListener("click",clicked);
window.requestAnimationFrame(update);
