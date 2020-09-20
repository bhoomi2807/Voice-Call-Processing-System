var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
// w = ctx.canvas.width = window.innerWidth;
// h = ctx.canvas.height = window.innerHeight;

w = ctx.canvas.width = 700;
  h = ctx.canvas.height = 100;

window.onresize = function() {
  // w = ctx.canvas.width = window.innerWidth;
  // h = ctx.canvas.height = window.innerHeight;
  w = ctx.canvas.width = 100;
  h = ctx.canvas.height = 100;
};

nt=0;
barWidth = 3;
gap = 4;
waveCount = 7;
waveScaleX = 180;
waveScaleY = 50;

var controls = new function() {
  this.barWidth = barWidth;
  this.gap = gap;
  this.waveCount = waveCount;
  this.waveScaleX = waveScaleX;
  this.waveScaleY = waveScaleY;
  
  this.redraw = function() {
    barWidth = controls.barWidth;
    gap = controls.gap;
    waveCount = controls.waveCount;
    waveScaleX = controls.waveScaleX;
    waveScaleY = controls.waveScaleY;
  }
}

// var gui = new dat.GUI({resizable : false});
// gui.add(controls, "barWidth", 1, 30).step(1).onChange(controls.redraw);
// gui.add(controls, "gap", 0, 30).step(1).onChange(controls.redraw);
// gui.add(controls, "waveCount", 1, 10).step(1).onChange(controls.redraw);
// gui.add(controls, "waveScaleX", 100, 1000).step(10).onChange(controls.redraw);
// gui.add(controls, "waveScaleY", 50, 300).step(10).onChange(controls.redraw);

function drawWave(n){
  nt+=0.01;
  for(i=0;i<n;i++){
    ctx.beginPath();
    for(x=0;x<w;x+=barWidth+gap){
      var y = noise.perlin3(x/waveScaleX, i, nt)*waveScaleY;
      ctx.fillStyle = "hsla("+(255+(20*i))+",100%,50%,0.1)";
      ctx.fillRect(x,(h/2),barWidth,y);
    }
    ctx.closePath();
  }
}

function clear(){
  ctx.fillStyle="rgba(10,5,30,0.06)";
  ctx.fillRect(0,0,w,h);
};

function render(){
  clear();
  drawWave(waveCount);
  requestAnimationFrame(render);
}
render();