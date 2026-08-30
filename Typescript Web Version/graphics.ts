export type Point = { x: number; y: number };

export type DrawCircleOptions = {
  fillStyle?: string | CanvasGradient | CanvasPattern;
  strokeStyle?: string | CanvasGradient | CanvasPattern;
  lineWidth?: number;
  /** Default true */
  fill?: boolean;
  /** Default false */
  stroke?: boolean;
};

export function drawCircle(
  ctx: CanvasRenderingContext2D,
  position: Point,
  radius: number,
  opts: DrawCircleOptions = {}
): void {
  if (radius <= 0){return;}
  const {
    fillStyle,
    strokeStyle,
    lineWidth,
    fill = true,
    stroke = false,
  } = opts;

  ctx.beginPath();
  ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);

  if (lineWidth !== undefined) ctx.lineWidth = lineWidth;
  if (fillStyle !== undefined) ctx.fillStyle = fillStyle;
  if (strokeStyle !== undefined) ctx.strokeStyle = strokeStyle;

  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}
