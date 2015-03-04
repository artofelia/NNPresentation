
var b = document.getElementById("b");
var c = document.getElementById("c");
var ctx = c.getContext('2d');
//var ic = document.getElementById("ic"); //for drawing input
//var ictx = ic.getContext('2d');

var numPts = (Object.keys(raw).length-1)/2;
var row_sz = raw['rowsz']; //pixels in row of image
var inn = raw['0x'].length; //number of pixels in image
var outn = 1; //number of outputs
var layer_size = [3,3]; //number of nuerons in each hidden layer
var L = layer_size.length; //number of hidden layers
var my_guess = [];
var iter = 0;

//console.log('inn',inn);

var xv = []; //value of neurons
var init_xv = function(){
	xv.push(Matrix.Zero(inn+1,1)); //constant in first row
	_.each(_.range(L), function(i){
		xv.push(Matrix.Zero(layer_size[i],1));
		});
	xv.push(Matrix.Zero(outn,1));
	my_guess.push(xv[L+1]);
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
		xv[0].elements[i+1][0] = parseInt(numstr[i]);
	});
	}else if(type=='xor'){
	_.each(_.range(numstr.length), function(i) {
		xv[0].elements[i+1][0] = parseInt(numstr[i])*2-1;
	});
	}
	//console.log('setin', xv[0].transpose().inspect());
}

var forward = function(ind){
	var numstr = raw[ind + 'x'];
	
	set_inputs(numstr, 'xor');
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
	return xv[L+1];
}

var format_output = function(yn, type) {
	var res = Matrix.Zero(outn, 1);
	if(type=='digit'){
		res.elements[yn][0] = 1;
	}else if(type=='xor'){
		res.elements[0][0] = yn;
	}
	return res;
}

var lrate = 0.1;//learning rate
var backprop = function(ind){
	var yn = raw[ind + 'y'];
	var y = format_output(yn, 'xor'); //given in form of network outputs
	//console.log('y',y.transpose().inspect());
	//set deltas
	
	//output deltas//var lastsub = my_guess[0].subtract(y);
	var lastsub = xv[L+1].subtract(y);
	console.log('error', lastsub.inspect());
	var lastcoeff = xv[L+1].map( function(x) {return x*x;} );
	del[L+1] = lastsub.map( function(x,i,j) {
		return 2 * x * (1-lastcoeff.e(i, j));
	});
	//console.log('delL+1', del[L+1].transpose().inspect());
	
	//hidden deltas
	_.each(_.range(L), function(sl){
		var l = L-sl; //going backwards
		//console.log('lind',l);
		var s = w[l+1].multiply(del[l+1]); //summation w[l][i][j]*del[l][j]
		var sem = s.transpose().elements[0];
		sem.splice(0,1);
		s = $M([sem]).transpose();
		var coeff = xv[l].map( function(x) {return x*x;} ); //for the derivative in chain rule
		console.log('coeff', coeff.inspect());
		var ndel = s.map( function(x, i, j){
			var nval = (1-coeff.e(i+1,j)) * s.e(i,j);
			//console.log(coeff.e(i+1,j), s.e(i,j), i, j);
			return nval; //calculate del[l-1][i]
		});
		//console.log('s', s.inspect());
		//console.log('ndel', ndel.inspect());
		del[l] = ndel;//remember first one is not used.
	});
	
	//update w
	_.each(_.range(L+1), function(ind) {
		var windch = ind+1;
		//console.log('wupind', windch);
		var ch = w[windch].map(function(x, i, j) {
			return lrate * xv[windch-1].e(i,1) * del[windch].e(j,1);
		});
		//console.log('ch',ch.inspect());
		w[windch] = w[windch].subtract(ch.dup());
		//console.log('chdim', ch.dimensions());
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
	var dy = 130;
	ctx.fillText("In:   " + raw[ind + 'x'], 30, dy);
	ctx.fillText("Out: " + raw[ind + 'y'], 30, dy+20);
	ctx.fillText("Guess: " + my_guess[0].e(1,1), 30, dy+40);
	ctx.fillText("Iter: " + iter, 30, dy+60);
}
//drawing variables
var drawNode = function(r, col, indj, indl) {
	ctx.beginPath();
	ctx.strokeStyle=col;
	var my_pos = node_pos[indl][indj];
	ctx.arc(my_pos[0],my_pos[1],r,0,2*Math.PI);
	ctx.stroke();
	ctx.closePath();
	ctx.fillStyle="black";
	//var xvval = xv[indl].e(indj+1,1);
	var xvval = Math.round(xv[indl].e(indj+1,1)*100)/100.0;
	ctx.fillText("" + xvval, my_pos[0]-15, my_pos[1]+3);
	if(indl != L+1){
		var rng = -1;
		var skp = -1;
		if(indl==L){
			_.each(_.range(outn), function(i) {
			var st = my_pos;
			var ed = node_pos[indl+1][i];
			drawAxon(st, ed, indj, i, indl+1);
			});
		}else {
			_.each(_.range(layer_size[indl]), function(i) {
			if(i==0) return;
			
			var st = my_pos;
			var ed = node_pos[indl+1][i];
			drawAxon(st, ed, indj, i-1, indl+1);
			});
		}
		
	}
}

var drawAxon = function(st, ed, indf, indt, indl) {
	ctx.beginPath();
	ctx.moveTo(st[0], st[1]);
	ctx.lineTo(ed[0], ed[1]);
	ctx.stroke();
	ctx.fillStyle="black";
	//console.log('axon');
	//console.log(indf,indt);
	var wval = Math.round(w[indl].e(indf+1,indt+1)*1000)/1000.0;
	var mu = 0.3;
	var tpos = [st[0]+(ed[0]-st[0])*mu,st[1]+(ed[1]-st[1])*mu];
	ctx.fillText("" + wval, tpos[0], tpos[1]);
}

var drawNetwork = function() {
	ctx.font="15px Georgia";
	var ra = 20;
	
	_.each(_.range(inn+1), function(i) {
		drawNode(ra, 'blue', i, 0);
	});
	
	_.each(_.range(L), function(l) {
		_.each(_.range(layer_size[l]), function(i) {
			drawNode(ra, 'green', i, l+1);
		});
	});
	_.each(_.range(outn), function(i) {
		drawNode(ra, 'blue', i, L+1);
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
b.onclick = function() {
	auto = !auto;
};
init_all(); //initialize weight, and xv arrays
var tind = 1;

var update = function() {
	ctx.fillStyle="#ffffff";
	ctx.fillRect(0,0,c.width,c.height);
	if(auto) {
		tind = Math.floor(numPts*Math.random());
		forward(tind);
		var retf = forward(tind);
		my_guess[0].elements[0][0] = Math.sign(retf.e(1,1));
		backprop(tind);
		iter++;
	}
	drawInOut(tind);
	drawNetwork();
	//drawNumber(tind);
	window.requestAnimationFrame(update);
}

var clicked = function(e) {
	var x = e.offsetX-300;
	var y = -e.offsetY+300;
	tind = Math.floor(numPts*Math.random());
	//console.log('tind', tind);
	var retf = forward(tind);
	my_guess[0].elements[0][0] = Math.sign(retf.e(1,1));
	backprop(tind);
	iter++;
	//console.log('update_out', w[1].inspect());
	
};

c.addEventListener("click",clicked);
window.requestAnimationFrame(update);
