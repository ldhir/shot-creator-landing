class FlickeringGrid {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.container = canvas.parentElement;

    // Options
    this.squareSize = options.squareSize || 4;
    this.gridGap = options.gridGap || 6;
    this.flickerChance = options.flickerChance || 0.3;
    this.color = options.color || 'rgb(255, 107, 122)';
    this.maxOpacity = options.maxOpacity || 0.3;

    // State
    this.isInView = false;
    this.animationFrameId = null;
    this.lastTime = 0;
    this.gridParams = null;

    // Convert color to RGBA format
    this.memoizedColor = this.toRGBA(this.color);

    this.init();
  }

  toRGBA(color) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = tempCanvas.height = 1;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return 'rgba(255, 107, 122,';
    tempCtx.fillStyle = color;
    tempCtx.fillRect(0, 0, 1, 1);
    const [r, g, b] = Array.from(tempCtx.getImageData(0, 0, 1, 1).data);
    return `rgba(${r}, ${g}, ${b},`;
  }

  setupCanvas(width, height) {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;

    const cols = Math.floor(width / (this.squareSize + this.gridGap));
    const rows = Math.floor(height / (this.squareSize + this.gridGap));

    const squares = new Float32Array(cols * rows);
    for (let i = 0; i < squares.length; i++) {
      squares[i] = Math.random() * this.maxOpacity;
    }

    return { cols, rows, squares, dpr };
  }

  updateSquares(squares, deltaTime) {
    for (let i = 0; i < squares.length; i++) {
      if (Math.random() < this.flickerChance * deltaTime) {
        squares[i] = Math.random() * this.maxOpacity;
      }
    }
  }

  drawGrid(cols, rows, squares, dpr) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        const opacity = squares[i * rows + j];
        this.ctx.fillStyle = `${this.memoizedColor}${opacity})`;
        this.ctx.fillRect(
          i * (this.squareSize + this.gridGap) * dpr,
          j * (this.squareSize + this.gridGap) * dpr,
          this.squareSize * dpr,
          this.squareSize * dpr
        );
      }
    }
  }

  updateCanvasSize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.gridParams = this.setupCanvas(width, height);
  }

  animate(time) {
    if (!this.isInView) return;

    const deltaTime = (time - this.lastTime) / 1000;
    this.lastTime = time;

    this.updateSquares(this.gridParams.squares, deltaTime);
    this.drawGrid(
      this.gridParams.cols,
      this.gridParams.rows,
      this.gridParams.squares,
      this.gridParams.dpr
    );

    this.animationFrameId = requestAnimationFrame((t) => this.animate(t));
  }

  init() {
    this.updateCanvasSize();

    // Resize observer
    const resizeObserver = new ResizeObserver(() => {
      this.updateCanvasSize();
    });
    resizeObserver.observe(this.container);

    // Intersection observer for performance
    const intersectionObserver = new IntersectionObserver(
      ([entry]) => {
        this.isInView = entry.isIntersecting;
        if (this.isInView && !this.animationFrameId) {
          this.animationFrameId = requestAnimationFrame((t) => this.animate(t));
        } else if (!this.isInView && this.animationFrameId) {
          cancelAnimationFrame(this.animationFrameId);
          this.animationFrameId = null;
        }
      },
      { threshold: 0 }
    );
    intersectionObserver.observe(this.canvas);

    this.resizeObserver = resizeObserver;
    this.intersectionObserver = intersectionObserver;
  }

  destroy() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    if (this.intersectionObserver) {
      this.intersectionObserver.disconnect();
    }
  }
}

// Initialize flickering grids on page load
document.addEventListener('DOMContentLoaded', () => {
  // Hero section
  const heroCanvas = document.getElementById('hero-canvas');
  if (heroCanvas) {
    heroCanvas.style.display = 'block';
    new FlickeringGrid(heroCanvas, {
      squareSize: 4,
      gridGap: 6,
      flickerChance: 0.3,
      color: 'rgb(255, 107, 122)',
      maxOpacity: 0.2
    });
  }

  // Other sections with canvas
  const sectionCanvases = document.querySelectorAll('.section-canvas');
  sectionCanvases.forEach((canvas) => {
    new FlickeringGrid(canvas, {
      squareSize: 4,
      gridGap: 6,
      flickerChance: 0.25,
      color: 'rgb(255, 107, 122)',
      maxOpacity: 0.15
    });
  });
});
