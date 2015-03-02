
var b = document.getElementById("b");
var c = document.getElementById("c");
var ctx = c.getContext('2d');
var ic = document.getElementById("ic"); //for drawing input
var ictx = ic.getContext('2d');

var numPts = Object.keys(raw).length;
var row_sz = raw['rowsz']; //pixels in row of image
var inn = raw['0x'].length; //number of pixels in image
var outn = 10; //number of outputs
var layer_size = [25, 15]; //number of nuerons in each hidden layer
var L = layer_size.length; //number of hidden layers

console.log('inn',inn);

var xv = []; //value of neurons
var init_xv = function(){
	xv.push(Matrix.Random(inn,1));
	_.each(_.range(L), function(i){
		xv.push(Matrix.Random(layer_size[i],1));
		});
	xv.push(Matrix.Random(outn,1));
	
}
init_xv();
console.log('xv',xv);

var w = [];
//matrix of weights between each layers of neurons
var init_w = function(){
	w.push([]);//index should start at 1
	var inp = Matrix.Random(inn, layer_size[0]-1);
	w.push(inp);
	var hidden = _.each(_.range(L-1), function(i) {
		w.push(Matrix.Random(layer_size[i], layer_size[i+1]-1));
	});
	var op = Matrix.Random(layer_size[L-1], outn);
	w.push(op);
}

init_w();
console.log('w',w);

var set_inputs = function(numstr){
	_.each(_.range(numstr.length), function(i) {
		xv[0].elements[i][0] = numstr[i];
	});
}

var forward = function(ind){
	var numstr = raw[ind + 'x'];
	set_inputs(numstr);
	_.each(_.range(L+2), function(l){ //each layer
		if (l==0) return;
		console.log(l, w[l].transpose().dimensions(), xv[l-1].dimensions());
		s = w[l].transpose().multiply(xv[l-1]);
		theta = s.map( function(x, i, j) {
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
var op = forward(0);
console.log(op.inspect(), op.dimensions());

var format_output = function(yn) {
	var res = Matrix.Zero(outn, 1);
	res.elements[0][yn] = 1;
	return res;
}

var del = [];

//del[a][b] - a=layer;b=neuron in layer, (a starts at 1), del[a] is a matrix as well 
var init_del = function(){
	del.push([]);
	_.each(_.range(L), function(i){
		del.push(Matrix.Random(layer_size[i],1));
		});
	del.push(Matrix.Random(outn,1));
}
init_del();

var backprop = function(yn){
	var y = format_output(yn); //given in form of network outputs
	
	//set deltas
	//final l = L
	_.each(_.range(outn), function(j) {
		del[L+1].elements[j] = 2 * (xv[L+1].e(j) - y[j]) * (1 - Math.pow(xv[L+1].e(j), 2));
	});
	_.each(_.range(L), function(sl){
		var l = L-sl; //moving backwards
		for (var i = 1; i < layer_size[l]; i++){ //first neuron is constant
			var sum = 0;
			for (var j = 0; j < layer_size[l+1]; j++){
				sum += w[l+1].e(i,j) * del[l+1].del(j);
			}
			del[l].elements[i] = (1 - Math.pow(xv[l].e(i), 2)) * sum;
		}
	}
	
}

var xyToScreen = function(pt){
	return [pt[0] + c.width/2.0, -pt[1] + c.height/2.0];
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
b.onclick = function() { auto = !auto };

var update = function() {
	ctx.fillStyle="#ffffff";
	ctx.fillRect(0,0,c.width,c.height);
	drawNumber(4);
	window.requestAnimationFrame(update);
}

var clicked = function(e) {
	var x = e.offsetX-300;
	var y = -e.offsetY+300;
	
};

c.addEventListener("click",clicked);
window.requestAnimationFrame(update);
